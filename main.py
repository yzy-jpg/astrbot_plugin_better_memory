# -*- coding: utf-8 -*-
"""更好的记忆 · Better Memory

AstrBot 插件：把"对话之外"的记忆做成一套分层认知系统。

- 情景记忆：每次对话都留一条带时间戳/情绪标签的小事件，会随时间衰减
- 语义记忆：自动抽取"我喜欢/我叫/我在…"这类事实，带信心值
- 人格记忆：可锁定的关系锚点 + 周期性复盘结论，每轮无条件注入
- 情感感知检索：score = 0.45*语义 + 0.20*鲜活 + 0.20*情绪一致 + 0.15*新近
- 动态遗忘：有效强度随时间指数衰减，被回忆会增强（记忆再巩固）
- 主动写入与复盘：规则抽取 + 可选 LLM 抽取 + 每 N 条消息自动复盘

命令：/mem （别名 /记忆）
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools

from .emotions import label_cn, sense
from .memory import (
    MemoryStore,
    extract_facts,
    is_milestone,
    parse_reflection_json,
    render_block,
    score_memories,
    short,
)

DEFAULTS = {
    "enabled": True,
    "inject_enabled": True,
    "episodic_enabled": True,
    "auto_write": True,
    "llm_extract": False,
    "reflect_enabled": True,
    "reflect_every": 60,
    "max_inject_chars": 1200,
    "top_k_episodic": 6,
    "top_k_semantic": 5,
    "half_life_days": 30.0,
    "min_episodic_importance": 0.30,
    "min_fact_importance": 0.45,
    "weights": {"semantic": 0.45, "vividness": 0.20, "emotional": 0.20, "recency": 0.15},
    "verbose": False,
}

HELP_TEXT = (
    "🧠 更好的记忆 · 命令\n"
    "/mem               查看记忆概览\n"
    "/mem core          查看人格/关系锚点\n"
    "/mem list [类型]   列出记忆（episodic/semantic/core）\n"
    "/mem add 内容      手动写入一条长期记忆\n"
    "/mem search 关键词 按关键词检索\n"
    "/mem pin 编号      把某条记忆锁定成核心（永不遗忘）\n"
    "/mem unpin 编号    解锁\n"
    "/mem forget 编号   删除某条记忆\n"
    "/mem reflect       立即复盘（需要可用模型）\n"
    "/mem export        导出当前记忆到文件\n"
    "/mem reset confirm 清空自己的记忆（危险）"
)


class Main(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self.conf = dict(DEFAULTS)
        if isinstance(config, dict):
            for key, value in config.items():
                if value is not None:
                    self.conf[key] = value
        self.data_dir = Path(StarTools.get_data_dir())
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = MemoryStore(self.data_dir / "memory.sqlite3")
        self._background: set[asyncio.Task] = set()
        self._cmd_cache: tuple[float, set[str]] | None = None
        logger.info("[better_memory] loaded, data dir: %s", self.data_dir)

    # ------------------------------------------------------------ 工具

    def _cfg(self, key: str, default=None):
        value = self.conf.get(key, DEFAULTS.get(key, default))
        return default if value is None else value

    def _scope(self, event: AstrMessageEvent) -> str:
        platform = event.get_platform_id() or "unknown"
        sender = event.get_sender_id() or "unknown"
        return f"{platform}:{sender}"

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _known_commands(self) -> set[str]:
        """当前已注册的命令名/别名（含插件自带的），用于识别并跳过命令消息。"""
        now = time.time()
        if self._cmd_cache and now - self._cmd_cache[0] < 30:
            return self._cmd_cache[1]
        names: set[str] = set()
        try:
            from astrbot.core.star.filter.command import CommandFilter
            from astrbot.core.star.star_handler import star_handlers_registry

            for handler in star_handlers_registry._handlers:
                for event_filter in handler.event_filters:
                    if isinstance(event_filter, CommandFilter):
                        names.add(event_filter.command_name)
                        names.update(event_filter.alias or set())
        except Exception:
            pass
        self._cmd_cache = (now, names)
        return names

    def _message_text(self, event: AstrMessageEvent) -> str:
        """取消息正文（唤醒前缀 / 命令前缀已被 AstrBot 去掉）。"""
        return (event.message_str or "").strip().lstrip("/／").strip()

    def _is_command(self, text: str) -> bool:
        tokens = text.split()
        return bool(tokens) and tokens[0] in self._known_commands()

    def _safe_md_name(self, scope: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_.-]", "_", scope)[:80]

    def _write_md(self, scope: str) -> Path:
        """把当前记忆导出成 MEMORY_<scope>.md，方便人类直接阅读/备份。"""
        stats = self.store.stats(scope)
        items = self.store.list(scope, limit=200)
        lines = [
            f"# 更好的记忆 · {scope}",
            "",
            f"- 导出时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 总计 {stats['total']} 条（情景 {stats['episodic']} / 语义 {stats['semantic']} / 核心 {stats['core']}），"
            f"复盘 {stats['reflections']} 条",
            "",
        ]
        groups = (
            ("人格 / 关系锚点", [m for m in items if m.get("pinned") or m["kind"] == "core"]),
            ("语义事实", [m for m in items if m["kind"] == "semantic" and not m.get("pinned")]),
            ("情景记忆", [m for m in items if m["kind"] == "episodic"]),
        )
        for title, group in groups:
            lines.append(f"## {title}（{len(group)}）")
            for m in group:
                when = time.strftime("%Y-%m-%d", time.localtime(float(m.get("created_at") or 0)))
                lines.append(
                    f"- #{m['id']} [{when}] {m['text']}"
                    f"（重要 {float(m.get('importance') or 0):.2f}，信心 {float(m.get('confidence') or 0):.2f}）"
                )
            lines.append("")
        reflections = self.store.reflections(scope, limit=30)
        if reflections:
            lines.append(f"## 关系复盘（{len(reflections)}）")
            for r in reflections:
                when = time.strftime("%Y-%m-%d", time.localtime(float(r.get("created_at") or 0)))
                lines.append(f"- [{when}] {r['text']}")
        path = self.data_dir / f"MEMORY_{self._safe_md_name(scope)}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # ------------------------------------------------------------ 写入

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def observe_message(self, event: AstrMessageEvent) -> None:
        """每条用户消息进来时：留情景记忆 + 主动抽取事实 + 计数/触发复盘。"""
        if not self._cfg("enabled", True):
            return
        text = self._message_text(event)
        if not text or self._is_command(text):
            return

        scope = self._scope(event)
        emo = sense(text)
        now = time.time()

        # 1) 情景记忆：一次交互一条小事件
        if self._cfg("episodic_enabled", True):
            importance = 0.35
            if is_milestone(text):
                importance = 0.8
            if emo["intensity"] > 0.35:
                importance = max(importance, 0.42 + 0.25 * float(emo["intensity"]))
            if len(text) > 60:
                importance += 0.05
            importance = min(1.0, importance)
            if importance >= float(self._cfg("min_episodic_importance", 0.30)):
                self.store.add(
                    scope,
                    "episodic",
                    text,
                    valence=emo["valence"],
                    arousal=emo["arousal"],
                    importance=importance,
                    confidence=0.9,
                    source="chat",
                )

        # 2) 语义/人格记忆：规则抽取（"我喜欢…" 这类）
        if self._cfg("auto_write", True):
            existing = {
                (m.get("text") or "").strip()
                for m in self.store.list(scope, limit=500)
            }
            for fact in extract_facts(text):
                if float(fact["importance"]) < float(self._cfg("min_fact_importance", 0.45)):
                    continue
                if fact["text"] in existing:
                    continue
                self.store.add(
                    scope,
                    fact["kind"],
                    fact["text"],
                    valence=emo["valence"],
                    arousal=emo["arousal"],
                    importance=fact["importance"],
                    confidence=fact["confidence"],
                    pinned=fact["kind"] == "core",
                    source="auto-extract",
                )
                existing.add(fact["text"])

        # 3) 可选：LLM 抽取（更懂语境，但有额外 token 成本）
        if self._cfg("llm_extract", False) and len(text) >= 6:
            self._spawn(self._llm_extract(scope, text))

        # 4) 计数 + 周期复盘
        count = self.store.bump_counter(scope)
        every = max(10, int(self._cfg("reflect_every", 60)))
        if self._cfg("reflect_enabled", True) and count % every == 0:
            self._spawn(self._reflect(scope))

    # ------------------------------------------------------------ 注入

    @filter.on_llm_request()
    async def inject_memories(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """请求 LLM 之前，把相关记忆拼进 System Prompt。"""
        if not self._cfg("enabled", True) or not self._cfg("inject_enabled", True):
            return
        if event.get_extra("_better_memory_injected"):
            return

        text = (event.message_str or "").strip()
        if not text:
            return

        scope = self._scope(event)
        try:
            candidates = self.store.candidates(scope, limit=400)
            if not candidates:
                return
            emo = sense(text)
            now = time.time()
            weights = self._cfg("weights") or {}
            half_life = float(self._cfg("half_life_days", 30.0))

            core = sorted(
                [m for m in candidates if m.get("pinned") or m["kind"] == "core"],
                key=lambda m: float(m.get("importance") or 0),
                reverse=True,
            )[:8]
            semantic = score_memories(
                [m for m in candidates if m["kind"] == "semantic" and not m.get("pinned")],
                text,
                emo,
                now,
                weights,
                half_life,
                int(self._cfg("top_k_semantic", 5)),
            )
            episodic = score_memories(
                [m for m in candidates if m["kind"] == "episodic"],
                text,
                emo,
                now,
                weights,
                half_life,
                int(self._cfg("top_k_episodic", 6)),
            )
            reflections = self.store.reflections(scope, limit=3)

            block = render_block(
                core,
                semantic,
                episodic,
                reflections,
                int(self._cfg("max_inject_chars", 1200)),
            )
            if not block:
                return
            req.system_prompt = (req.system_prompt or "") + block
            event.set_extra("_better_memory_injected", True)
            self.store.touch([int(m["id"]) for m in semantic + episodic])
            if self._cfg("verbose", False):
                logger.info(
                    "[better_memory] injected %s core / %s semantic / %s episodic (chars=%s)",
                    len(core), len(semantic), len(episodic), len(block),
                )
        except Exception:
            logger.exception("[better_memory] failed to inject memories")

    # ------------------------------------------------------------ LLM 辅助

    async def _provider(self):
        try:
            provider = self.context.get_using_provider()
        except Exception:
            provider = None
        if provider is None:
            try:
                provider = await self.context.get_using_provider_async()
            except Exception:
                provider = None
        return provider

    async def _llm_extract(self, scope: str, text: str) -> None:
        """用 LLM 判断这条消息里有没有值得长期记住的事实（带信心值）。"""
        provider = await self._provider()
        if provider is None:
            return
        prompt = (
            "你是记忆整理助手。从下面这句话里提取值得长期记住的用户事实（偏好/身份/关系/目标/红线）。\n"
            "要求：只输出 JSON，格式 {\"facts\":[{\"text\":\"…\",\"confidence\":0.0-1.0,\"importance\":0.0-1.0}]}；"
            "没有就输出 {\"facts\":[]}。每条 30 字以内，不要编造。\n\n"
            f"用户说：{short(text, 300)}"
        )
        try:
            resp = await provider.text_chat(
                prompt=prompt, system_prompt="你只输出严格 JSON。", contexts=[]
            )
            data = parse_reflection_json(getattr(resp, "completion_text", "") or "")
            existing = {(m.get("text") or "").strip() for m in self.store.list(scope, limit=500)}
            for fact in data.get("facts", []) or []:
                value = str(fact.get("text") or "").strip()
                if not value or value in existing or len(value) > 80:
                    continue
                self.store.add(
                    scope,
                    "semantic",
                    value,
                    importance=min(0.95, float(fact.get("importance") or 0.6)),
                    confidence=min(0.95, float(fact.get("confidence") or 0.6)),
                    source="llm-extract",
                )
                existing.add(value)
        except Exception:
            logger.exception("[better_memory] llm extract failed")

    async def _reflect(self, scope: str) -> None:
        """周期性复盘：从记忆里提炼新的人格/关系规则，写回核心记忆。"""
        provider = await self._provider()
        if provider is None:
            return
        items = self.store.list(scope, limit=60)
        if not items:
            return
        memories = "\n".join(
            f"- [{m['kind']}] {short(m['text'], 100)}" for m in items
        )
        prompt = (
            "你在帮一个 AI 伴侣做「记忆复盘」。下面是它和用户最近的记忆条目。\n"
            "请提炼出：(1) 稳定的用户画像/偏好 (2) 关系规则或应长期遵守的约定。\n"
            "只输出 JSON：{\"core\":[{\"text\":\"…\"}],\"facts\":[{\"text\":\"…\"}]}；"
            "各不超过 3 条，每条 40 字以内，用第二人称对 AI 说话的口吻（例如\"用户喜欢…，要多留意\"）。"
            "不要编造记忆里没有的内容。\n\n" + memories
        )
        try:
            resp = await provider.text_chat(
                prompt=prompt, system_prompt="你只输出严格 JSON。", contexts=[]
            )
            data = parse_reflection_json(getattr(resp, "completion_text", "") or "")
            new_core = [str(x.get("text") or "").strip() for x in (data.get("core") or []) if x]
            new_facts = [str(x.get("text") or "").strip() for x in (data.get("facts") or []) if x]
            existing = {(m.get("text") or "").strip() for m in self.store.list(scope, limit=500)}
            for text in new_core:
                if text and text not in existing and len(text) <= 120:
                    self.store.add(
                        scope, "core", text, importance=0.9, confidence=0.75,
                        pinned=True, source="reflection",
                    )
                    existing.add(text)
            for text in new_facts:
                if text and text not in existing and len(text) <= 120:
                    self.store.add(
                        scope, "semantic", text, importance=0.7, confidence=0.7,
                        source="reflection",
                    )
                    existing.add(text)
            if new_core or new_facts:
                summary = "复盘：" + "；".join((new_core + new_facts)[:4])
                self.store.add_reflection(scope, summary)
                self.store.mark_reflected(scope)
                self._write_md(scope)
                logger.info("[better_memory] reflected %s new items for %s", len(new_core) + len(new_facts), scope)
        except Exception:
            logger.exception("[better_memory] reflection failed")

    # ------------------------------------------------------------ 命令

    @filter.command("mem", alias={"记忆"})
    async def mem(self, event: AstrMessageEvent) -> None:
        scope = self._scope(event)
        raw = (event.message_str or "").strip()
        if raw[:1] in ("/", "／"):
            raw = raw[1:].strip()
        tokens = raw.split()
        if tokens and tokens[0] in ("mem", "记忆"):
            tokens = tokens[1:]
        sub = tokens[0].lower() if tokens else ""
        rest = " ".join(tokens[1:]).strip() if len(tokens) > 1 else ""

        if sub in ("", "help", "帮助"):
            stats = self.store.stats(scope)
            yield event.plain_result(
                f"🧠 记忆概览（{scope}）\n"
                f"- 情景 {stats['episodic']} 条 / 语义 {stats['semantic']} 条 / 核心 {stats['core']} 条\n"
                f"- 已锁定 {stats['pinned']} 条，复盘 {stats['reflections']} 条，累计观察 {stats['messages']} 条消息\n\n"
                + HELP_TEXT
            )
            return

        if sub == "core":
            items = [m for m in self.store.list(scope, limit=200) if m.get("pinned") or m["kind"] == "core"]
            if not items:
                yield event.plain_result("还没有锁定的核心记忆。用 /mem pin 编号 把重要的事钉住，它会每轮都注入。")
                return
            lines = [f"🔒 核心记忆（{len(items)} 条，永不衰减）"]
            for m in items[:20]:
                lines.append(f"#{m['id']} {short(m['text'], 90)}")
            yield event.plain_result("\n".join(lines))
            return

        if sub == "list":
            kind = rest.lower() if rest else None
            items = self.store.list(scope, kind=kind, limit=20)
            if not items:
                yield event.plain_result("没有找到记忆。")
                return
            lines = [f"📚 记忆列表（{kind or '全部'}，最近 {len(items)} 条）"]
            for m in items:
                when = time.strftime("%m-%d", time.localtime(float(m.get("created_at") or 0)))
                mark = "🔒" if m.get("pinned") else ""
                lines.append(f"#{m['id']} {mark}[{m['kind']} {when}] {short(m['text'], 70)}")
            yield event.plain_result("\n".join(lines))
            return

        if sub == "add":
            if not rest:
                yield event.plain_result("用法：/mem add 想记住的内容")
                return
            existing = {(m.get("text") or "").strip() for m in self.store.list(scope, limit=500)}
            if rest in existing:
                yield event.plain_result("这条已经记过了。")
                return
            facts = extract_facts(rest)
            if facts:
                fact = facts[0]
                kind = fact["kind"]
                importance = fact["importance"]
                confidence = fact["confidence"]
            else:
                kind, importance, confidence = "semantic", 0.8, 0.95
            mem_id = self.store.add(
                scope,
                kind,
                rest,
                importance=importance,
                confidence=confidence,
                pinned=kind == "core",
                source="manual",
            )
            self._write_md(scope)
            label = {"core": "（核心记忆，永不遗忘）", "semantic": "", "episodic": ""}[kind]
            yield event.plain_result(f"已记住 #{mem_id}{label}：{short(rest, 80)}")
            return

        if sub == "search":
            if not rest:
                yield event.plain_result("用法：/mem search 关键词")
                return
            emo = sense(rest)
            ranked = score_memories(
                self.store.candidates(scope, limit=400),
                rest,
                emo,
                time.time(),
                self._cfg("weights") or {},
                float(self._cfg("half_life_days", 30.0)),
                8,
            )
            if not ranked:
                yield event.plain_result("没有找到相关记忆。")
                return
            lines = [f"🔎 与「{short(rest, 20)}」最相关的记忆"]
            for m in ranked:
                lines.append(f"#{m['id']}（{m['_score']:.2f}）{short(m['text'], 80)}")
            yield event.plain_result("\n".join(lines))
            return

        if sub in ("pin", "unpin", "forget"):
            if not rest.isdigit():
                yield event.plain_result(f"用法：/mem {sub} 记忆编号（编号可用 /mem list 查看）")
                return
            mem_id = int(rest)
            if self.store.get(mem_id) is None:
                yield event.plain_result(f"没有找到编号 {mem_id} 的记忆。")
                return
            if sub == "forget":
                self.store.delete(mem_id)
                self._write_md(scope)
                yield event.plain_result(f"已删除记忆 #{mem_id}。")
            else:
                ok = self.store.pin(mem_id, pinned=(sub == "pin"))
                yield event.plain_result(
                    f"已{'锁定' if sub == 'pin' else '解锁'}记忆 #{mem_id}。" if ok else "操作失败。"
                )
            return

        if sub == "reflect":
            yield event.plain_result("开始复盘，完成后会把新的核心记忆写回…")
            await self._reflect(scope)
            yield event.plain_result("复盘完成。用 /mem core 查看新的核心记忆。")
            return

        if sub == "export":
            data = self.store.export(scope)
            path = self.data_dir / f"export_{self._safe_md_name(scope)}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            md = self._write_md(scope)
            yield event.plain_result(f"已导出：\n{path}\n{md}")
            return

        if sub == "reset":
            if rest != "confirm":
                yield event.plain_result("确认清空请发送：/mem reset confirm（会删除你自己的全部记忆，不可恢复）")
                return
            self.store.reset(scope)
            self._write_md(scope)
            yield event.plain_result("已清空你的记忆。")
            return

        yield event.plain_result("未知子命令。\n" + HELP_TEXT)

    # ------------------------------------------------------------ 生命周期

    async def terminate(self) -> None:
        for task in list(self._background):
            task.cancel()
        try:
            self.store.close()
        except Exception:
            pass
        logger.info("[better_memory] unloaded")
