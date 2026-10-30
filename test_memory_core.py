# -*- coding: utf-8 -*-
"""离线自测：emotions / memory 两个模块的核心逻辑（不需要 AstrBot 运行）。"""
import sys, time, tempfile, os, json
PLUGIN = r"C:\Users\Jazy Yang\WorkBuddy\2026-10-15-17-18-35\qqbot\plugins\astrbot_plugin_better_memory"
sys.path.insert(0, PLUGIN)
import emotions
import memory as mem

print("== 情绪感知 ==")
for t in ["今天好开心啊，谢谢你陪我！！😊", "我好难过，感觉要崩溃了……", "我明天要上班", "我不喜欢下雨天"]:
    r = emotions.sense(t)
    print(f"  {t!r:40} -> valence={r['valence']:+.2f} arousal={r['arousal']:.2f} label={emotions.label_cn(r['valence'], r['arousal'])}")

print("== 事实抽取 ==")
for t in ["我叫小杨，今年 20 岁", "我喜欢下雨天和猫", "我讨厌香菜", "记住我下周三考试", "我住在杭州", "我在字节上班"]:
    print("  ", t, "->", [f["text"] for f in mem.extract_facts(t)])

print("== 存储 / 检索 ==")
tmp = tempfile.mkdtemp()
store = mem.MemoryStore(os.path.join(tmp, "m.sqlite3"))
scope = "qq:FriendMessage:test"
now = time.time()
store.add(scope, "episodic", "第一次说我爱你", valence=0.8, arousal=0.7, importance=0.9, source="test")
store.add(scope, "semantic", "喜欢下雨天和猫", valence=0.5, arousal=0.2, importance=0.7, confidence=0.9)
store.add(scope, "episodic", "今天加班到很晚，很累", valence=-0.6, arousal=0.5, importance=0.6)
store.add(scope, "core", "用户是重要的人，不能敷衍", importance=0.95, pinned=True)
# 造一条很老的记忆
old = store.add(scope, "episodic", "很久以前的事", importance=0.9)
store._conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (now - 400 * 86400, old))
store._conn.commit()

cands = store.candidates(scope, 100)
core = [m for m in cands if m.get("pinned") or m["kind"] == "core"]
epi = [m for m in cands if m["kind"] == "episodic"]
print("  候选:", len(cands), "核心:", len(core), "情景:", len(epi))
# 难过时检索，应优先负向记忆
emo_q = emotions.sense("我今天好难过啊")
ranked = mem.score_memories(epi, "我今天好难过啊", emo_q, now, {}, 30.0, 3)
for m in ranked:
    print(f"    score={m['_score']:.3f} sim={m['_sim']:.2f} strength={m['_strength']:.2f} v={m['valence']:+.1f} {m['text']}")
assert ranked, "检索结果为空"
# 老记忆应被衰减
old_ranked = mem.score_memories([m for m in epi if m["text"] == "很久以前的事"], "很久以前的事", emo_q, now, {}, 30.0, 1)
print("  400 天前记忆有效强度:", old_ranked[0]["_strength"] if old_ranked else "被衰减掉（strength<0.02）")

block = mem.render_block(core, [m for m in cands if m["kind"] == "semantic"], ranked, store.reflections(scope, 3), 1200)
print("== 注入块长度 ==", len(block))
print(block[:400])

# touch / pin / reset
store.touch([core[0]["id"]])
print("  touch 后 strength:", store.get(core[0]["id"])["strength"], "access:", store.get(core[0]["id"])["access_count"])
print("  stats:", store.stats(scope))
print("== 全部通过 ==")
