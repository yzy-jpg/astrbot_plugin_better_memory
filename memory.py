# -*- coding: utf-8 -*-
"""分层记忆的存储、检索、写入与复盘逻辑。

四层记忆（与认知科学的分层对应）：

| 层 | kind | 特点 |
| --- | --- | --- |
| 工作记忆 | 不落库 | 即 AstrBot 的对话上下文窗口 |
| 情景记忆 | episodic | 每次对话的带时间戳/情绪标签的小事件，会衰减 |
| 语义记忆 | semantic | 从对话里抽出来的事实与偏好，相对稳定 |
| 人格记忆 | core | 锁定的关系锚点/人格设定/复盘结论，永不衰减，每轮必注入 |

检索评分（可配置权重）：

    score = w_sem * 语义相似度(BM25)
          + w_viv * 鲜活度(重要性/复述次数/强度)
          + w_emo * 情绪一致性
          + w_rec * 新近度

再乘上"有效强度"（随时间指数衰减，pinned 免衰减）。
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

STOPWORDS = set(
    """
的 了 是 我 你 他 她 它 们 这 那 有 在 和 与 就 都 也 又 还 被 把 给 对 吗 呢 吧 啊
呀 哦 嗯 哈 啦 着 过 我们 你们 他们 一个 一下 什么 怎么 为什么 因为 所以 但是 而且
如果 然后 现在 就是 可以 没有 不是 这个 那个 自己 觉得 感觉 知道 时候 一些 这样 那样
""".split()
)

MEMORY_KINDS = ("episodic", "semantic", "core")

# ---------------------------------------------------------------- 文本工具


def tokenize(text: str) -> list[str]:
    """jieba 分词 + 去停用词，用于 BM25 相似度。"""
    words = []
    for w in jieba.lcut((text or "").lower()):
        w = w.strip()
        if not w or w in STOPWORDS:
            continue
        if re.fullmatch(r"[\W_]+", w):
            continue
        words.append(w)
    return words


def keywords_of(text: str, limit: int = 24) -> str:
    """从文本里抽关键词（去停用词后按频次/词长粗排），存库用于召回。"""
    seen: dict[str, int] = {}
    for w in tokenize(text):
        if len(w) < 2 and not re.fullmatch(r"[\u4e00-\u9fff]", w):
            continue
        seen[w] = seen.get(w, 0) + 1
    ranked = sorted(seen.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    return " ".join(w for w, _ in ranked[:limit])


def short(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text if len(text) <= limit else text[:limit] + "…"


# ---------------------------------------------------------------- 事实抽取

# (正则, kind, 置信度, 重要性)
FACT_PATTERNS = (
    (r"我叫([^\s，。,.!！?？]{1,20})", "semantic", 0.95, 0.85),
    (r"我的名字(?:是|叫)([^\s，。,.!！?？]{1,20})", "semantic", 0.95, 0.85),
    (r"我(?:住在|家在|来自)([^。！？!?，,]{1,20})", "semantic", 0.9, 0.7),
    (r"我(?:的)?生日(?:是)?([^。！？!?，,]{1,20})", "core", 0.95, 0.95),
    (r"我(?:喜欢|爱|最爱|迷上了?)([^。！？!?，,]{1,30})", "semantic", 0.9, 0.75),
    (r"我(?:不喜欢|讨厌|受不了|很反感)([^。！？!?，,]{1,30})", "semantic", 0.85, 0.7),
    (r"我(?:想要|想买|希望)([^。！？!?，,]{1,30})", "semantic", 0.7, 0.5),
    (r"我(?:在)([^。！？!?，,]{1,20})(?:工作|上班|上学|读书|念书)", "semantic", 0.8, 0.7),
    (r"我(?:害怕|怕)([^。！？!?，,]{1,30})", "semantic", 0.8, 0.6),
    (r"(?:记住|别忘|不要忘|记一下)([^。！？!?]{1,60})", "semantic", 0.9, 0.85),
)

_MILESTONE_RE = re.compile(r"第一次|纪念日|周年|生日|在一起|表白|毕业|入职|搬家|旅行")


def extract_facts(text: str) -> list[dict]:
    """从一条用户消息里抽取值得长期记住的事实。"""
    text = short(text, 300)
    facts: list[dict] = []
    for pattern, kind, confidence, importance in FACT_PATTERNS:
        for m in re.finditer(pattern, text):
            value = (m.group(1) or "").strip("，。,.!！?？ ")
            if not value:
                continue
            facts.append(
                {
                    "kind": kind,
                    "text": m.group(0).strip(),
                    "confidence": confidence,
                    "importance": importance,
                }
            )
    unique: dict[str, dict] = {}
    for f in facts:
        unique.setdefault(f["text"], f)
    return list(unique.values())


def is_milestone(text: str) -> bool:
    return bool(_MILESTONE_RE.search(text or ""))


# ---------------------------------------------------------------- 检索打分


def _effective_strength(mem: dict, now: float, half_life_days: float) -> float:
    if mem.get("pinned"):
        return 1.0
    age_days = max(0.0, (now - float(mem.get("created_at") or now)) / 86400.0)
    decay = 0.5 ** (age_days / max(half_life_days, 0.5))
    return max(0.0, float(mem.get("strength") or 1.0) * decay)


def _emotional_congruence(mem: dict, q_emo: dict) -> float:
    """情绪一致性：开心时更容易想起开心的事，难过时更容易想起难过的事。"""
    q_int = float(q_emo.get("intensity") or 0.0)
    if q_int < 0.12:
        return 0.5
    qv = float(q_emo.get("valence") or 0.0)
    qa = float(q_emo.get("arousal") or 0.0)
    mv = float(mem.get("valence") or 0.0)
    ma = float(mem.get("arousal") or 0.0)
    valence_match = 1.0 - abs(qv - mv) / 2.0
    arousal_match = 1.0 - abs(qa - ma)
    score = 0.7 * valence_match + 0.3 * arousal_match
    return min(1.0, score * (0.85 + 0.15 * min(1.0, abs(mv) + ma)))


def score_memories(
    memories: list[dict],
    query: str,
    q_emo: dict,
    now: float,
    weights: dict,
    half_life_days: float,
    limit: int,
) -> list[dict]:
    """对候选记忆打分排序，返回 top-limit（带 _score/_strength 字段）。"""
    if not memories:
        return []

    tokens = tokenize(query)
    sims = [0.0] * len(memories)
    if tokens:
        corpus = [tokenize((m.get("text") or "") + " " + (m.get("keywords") or "")) for m in memories]
        if any(corpus):
            try:
                bm25 = BM25Okapi(corpus)
                sims = [max(0.0, float(s)) for s in bm25.get_scores(tokens)]
            except Exception:
                sims = [0.0] * len(memories)
    max_sim = max(sims) if sims else 0.0
    sims = [s / max_sim if max_sim > 0 else 0.0 for s in sims]

    w = {
        "semantic": float(weights.get("semantic", 0.45)),
        "vividness": float(weights.get("vividness", 0.20)),
        "emotional": float(weights.get("emotional", 0.20)),
        "recency": float(weights.get("recency", 0.15)),
    }

    scored: list[dict] = []
    for mem, sim in zip(memories, sims):
        strength = _effective_strength(mem, now, half_life_days)
        if strength <= 0.02:
            continue
        age_days = max(0.0, (now - float(mem.get("created_at") or now)) / 86400.0)
        recency = 0.5 ** (age_days / max(half_life_days, 0.5))
        access = min(1.0, float(mem.get("access_count") or 0) / 5.0)
        vividness = 0.45 * float(mem.get("importance") or 0.4) + 0.25 * access + 0.30 * strength
        emotional = _emotional_congruence(mem, q_emo)
        score = (
            w["semantic"] * sim
            + w["vividness"] * vividness
            + w["emotional"] * emotional
            + w["recency"] * recency
        )
        if mem.get("pinned"):
            score += 0.35
        item = dict(mem)
        item["_score"] = round(score, 4)
        item["_sim"] = round(sim, 4)
        item["_strength"] = round(strength, 4)
        scored.append(item)

    scored.sort(key=lambda m: m["_score"], reverse=True)
    return scored[: max(1, int(limit))]


# ---------------------------------------------------------------- 存储


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    keywords TEXT NOT NULL DEFAULT '',
    valence REAL NOT NULL DEFAULT 0,
    arousal REAL NOT NULL DEFAULT 0,
    importance REAL NOT NULL DEFAULT 0.4,
    confidence REAL NOT NULL DEFAULT 0.6,
    pinned INTEGER NOT NULL DEFAULT 0,
    strength REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    last_access REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope, kind);
CREATE INDEX IF NOT EXISTS idx_mem_created ON memories(scope, created_at);

CREATE TABLE IF NOT EXISTS reflections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ref_scope ON reflections(scope);

CREATE TABLE IF NOT EXISTS counters (
    scope TEXT PRIMARY KEY,
    msg_count INTEGER NOT NULL DEFAULT 0,
    last_reflect REAL NOT NULL DEFAULT 0
);
"""


class MemoryStore:
    """SQLite 记忆库（WAL 模式，线程安全）。"""

    def __init__(self, db_path: Path | str) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- 基础读取 ----------------------------------------------------
    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]

    def get(self, mem_id: int) -> dict | None:
        rows = self._rows("SELECT * FROM memories WHERE id = ?", (int(mem_id),))
        return rows[0] if rows else None

    def list(self, scope: str, kind: str | None = None, limit: int = 100) -> list[dict]:
        if kind in MEMORY_KINDS:
            return self._rows(
                "SELECT * FROM memories WHERE scope = ? AND kind = ? "
                "ORDER BY pinned DESC, created_at DESC LIMIT ?",
                (scope, kind, int(limit)),
            )
        return self._rows(
            "SELECT * FROM memories WHERE scope = ? ORDER BY pinned DESC, created_at DESC LIMIT ?",
            (scope, int(limit)),
        )

    def candidates(self, scope: str, limit: int = 400) -> list[dict]:
        return self._rows(
            "SELECT * FROM memories WHERE scope = ? ORDER BY created_at DESC LIMIT ?",
            (scope, int(limit)),
        )

    def reflections(self, scope: str, limit: int = 20) -> list[dict]:
        return self._rows(
            "SELECT * FROM reflections WHERE scope = ? ORDER BY created_at DESC LIMIT ?",
            (scope, int(limit)),
        )

    def stats(self, scope: str) -> dict:
        out = {"total": 0, "episodic": 0, "semantic": 0, "core": 0, "pinned": 0, "reflections": 0}
        for row in self._rows(
            "SELECT kind, COUNT(*) AS n FROM memories WHERE scope = ? GROUP BY kind", (scope,)
        ):
            kind = row["kind"]
            out[kind] = int(row["n"])
            out["total"] += int(row["n"])
        pinned = self._rows(
            "SELECT COUNT(*) AS n FROM memories WHERE scope = ? AND pinned = 1", (scope,)
        )
        out["pinned"] = int(pinned[0]["n"]) if pinned else 0
        refs = self._rows("SELECT COUNT(*) AS n FROM reflections WHERE scope = ?", (scope,))
        out["reflections"] = int(refs[0]["n"]) if refs else 0
        out["messages"] = self.counter(scope)[0]
        return out

    # -- 写入 --------------------------------------------------------
    def add(
        self,
        scope: str,
        kind: str,
        text: str,
        *,
        valence: float = 0.0,
        arousal: float = 0.0,
        importance: float = 0.4,
        confidence: float = 0.6,
        pinned: bool = False,
        source: str = "auto",
    ) -> int:
        now = time.time()
        if kind not in MEMORY_KINDS:
            kind = "semantic"
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories (scope, kind, text, keywords, valence, arousal, importance,"
                " confidence, pinned, strength, created_at, last_access, access_count, source)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    scope,
                    kind,
                    short(text, 400),
                    keywords_of(text),
                    float(valence),
                    float(arousal),
                    float(importance),
                    float(confidence),
                    1 if pinned else 0,
                    1.0,
                    now,
                    now,
                    0,
                    source,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def touch(self, mem_ids: list[int], now: float | None = None) -> None:
        """被回忆：强度增强 + 访问计数 + 记忆再巩固（关键词轻微漂移）。"""
        if not mem_ids:
            return
        now = now or time.time()
        with self._lock:
            for mem_id in mem_ids:
                row = self._conn.execute(
                    "SELECT strength, access_count, keywords FROM memories WHERE id = ?", (int(mem_id),)
                ).fetchone()
                if not row:
                    continue
                strength = min(1.0, float(row["strength"]) + 0.06)
                self._conn.execute(
                    "UPDATE memories SET strength = ?, access_count = access_count + 1, last_access = ? "
                    "WHERE id = ?",
                    (strength, now, int(mem_id)),
                )
            self._conn.commit()

    def pin(self, mem_id: int, pinned: bool = True) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE memories SET pinned = ?, kind = CASE WHEN ? THEN 'core' ELSE kind END WHERE id = ?",
                (1 if pinned else 0, 1 if pinned else 0, int(mem_id)),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete(self, mem_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (int(mem_id),))
            self._conn.commit()
            return cur.rowcount > 0

    def add_reflection(self, scope: str, text: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO reflections (scope, text, created_at) VALUES (?,?,?)",
                (scope, short(text, 600), time.time()),
            )
            self._conn.commit()

    def reset(self, scope: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE scope = ?", (scope,))
            self._conn.execute("DELETE FROM reflections WHERE scope = ?", (scope,))
            self._conn.commit()

    def export(self, scope: str) -> dict:
        return {
            "scope": scope,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "memories": self.list(scope, limit=5000),
            "reflections": self.reflections(scope, limit=500),
        }

    # -- 计数器 ------------------------------------------------------
    def counter(self, scope: str) -> tuple[int, float]:
        rows = self._rows("SELECT msg_count, last_reflect FROM counters WHERE scope = ?", (scope,))
        if not rows:
            return 0, 0.0
        return int(rows[0]["msg_count"]), float(rows[0]["last_reflect"])

    def bump_counter(self, scope: str) -> int:
        with self._lock:
            self._conn.execute(
                "INSERT INTO counters (scope, msg_count, last_reflect) VALUES (?, 1, 0)"
                " ON CONFLICT(scope) DO UPDATE SET msg_count = msg_count + 1",
                (scope,),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT msg_count FROM counters WHERE scope = ?", (scope,)).fetchone()
            return int(row[0]) if row else 0

    def mark_reflected(self, scope: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO counters (scope, msg_count, last_reflect) VALUES (?, 0, ?)"
                " ON CONFLICT(scope) DO UPDATE SET last_reflect = excluded.last_reflect",
                (scope, time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------- 记忆块渲染


def _fmt_time(ts: float) -> str:
    try:
        return time.strftime("%m-%d", time.localtime(float(ts or 0)))
    except Exception:
        return "--"


def render_block(
    core_items: list[dict],
    semantic_items: list[dict],
    episodic_items: list[dict],
    reflection_items: list[dict],
    max_chars: int,
) -> str:
    """把四层记忆渲染成注入 System Prompt 的文本块（带字符预算）。"""
    lines: list[str] = []
    budget = max(200, int(max_chars))

    def add_section(title: str, items: list[dict], fmt) -> None:
        nonlocal budget
        if not items:
            return
        section = [f"### {title}"]
        for item in items:
            line = fmt(item)
            if budget - len(line) - 1 < 0:
                section.append("- …（更多记忆已省略）")
                break
            section.append(line)
            budget -= len(line) + 1
        if len(section) > 1:
            lines.extend(section)

    add_section(
        "人格与关系锚点（永久，必须遵守）",
        core_items,
        lambda m: f"- {short(m['text'], 80)}",
    )
    add_section(
        "已知事实（语义记忆）",
        semantic_items,
        lambda m: f"- {short(m['text'], 80)}（信心 {int(float(m.get('confidence') or 0) * 100)}%）",
    )
    add_section(
        "相关往事（情景记忆）",
        episodic_items,
        lambda m: f"- [{_fmt_time(m.get('created_at'))} · 情绪 {int(float(m.get('valence') or 0) * 100):+d}] "
        f"{short(m['text'], 90)}",
    )
    add_section(
        "关系复盘（长期修正）",
        reflection_items,
        lambda m: f"- {short(m['text'], 100)}",
    )

    if not lines:
        return ""
    return (
        "\n\n【分层记忆 · 关于眼前这个人（自然使用，不要逐条复述，也不要暴露数据库结构）】\n"
        + "\n".join(lines)
        + "\n【记忆结束】"
    )


def parse_reflection_json(raw: str) -> dict:
    """从 LLM 回复里尽力解析出 JSON（容忍 json 代码块包裹/前后废话）。"""
    raw = (raw or "").strip()
    if not raw:
        return {}
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
