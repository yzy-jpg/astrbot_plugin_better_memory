# -*- coding: utf-8 -*-
"""轻量中文情绪感知。

不依赖任何外部模型，用词表 + 否定词 + 标点 + 表情符号估计一条消息的情绪状态：

- valence  : 愉悦度，[-1, 1]，负值代表消极
- arousal  : 唤醒度，[0, 1]，越高代表情绪越激动
- intensity: 整体强度，[0, 1]，用于决定"这次对话值不值得记一笔"

全部为纯函数，方便测试与复用。
"""

from __future__ import annotations

import math
import re

POSITIVE_WORDS = (
    "开心", "高兴", "快乐", "喜欢", "爱", "太好了", "好耶", "谢谢", "感谢", "棒",
    "厉害", "可爱", "温柔", "幸福", "舒服", "满足", "期待", "惊喜", "感动", "甜",
    "哈哈", "笑", "美", "帅", "优秀", "顺利", "成功", "赞", "完美", "温暖", "放心",
    "安心", "治愈", "心动", "抱抱", "么么", "亲亲", "真好", "不错", "好棒", "爱你",
    "想你", "美好", "幸运", "欣慰", "踏实", "享受", "兴奋", "骄傲", "舒服",
)

NEGATIVE_WORDS = (
    "难过", "伤心", "生气", "愤怒", "讨厌", "烦", "累", "疲惫", "焦虑", "害怕",
    "担心", "孤独", "委屈", "崩溃", "痛苦", "失望", "糟糕", "哭", "郁闷", "压力",
    "失眠", "难受", "心累", "无语", "恶心", "恐惧", "不安", "沮丧", "灰心", "绝望",
    "寂寞", "想哭", "冷淡", "敷衍", "烦人", "破防", "内耗", "压抑", "烦躁", "疼",
    "生病", "熬夜", "掉头发", "被骂", "倒霉", "崩溃", "气死",
)

INTENSIFIERS = ("很", "非常", "特别", "超级", "太", "真的", "巨", "极其", "十分", "超", "贼", "相当", "无敌")

NEGATIONS = ("不", "没", "没有", "别", "无", "未")

POSITIVE_EMOJI = ("😀", "😃", "😄", "😁", "😆", "😊", "😍", "🥰", "😘", "😌", "😎", "🥳", "🤗", "❤", "💕", "✨", "👍", "🎉", "😇", "🙂")
NEGATIVE_EMOJI = ("😢", "😭", "😔", "😞", "😡", "🤬", "😰", "😨", "😩", "😫", "💔", "😖", "😣", "🙁", "😒")

_AROUSAL_PUNCT = re.compile(r"[!！]+|[?？]{2,}")
_LAUGH = re.compile(r"[哈嘿嘻]{2,}")


def _positions(text: str, words) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for w in words:
        start = 0
        while True:
            idx = text.find(w, start)
            if idx < 0:
                break
            hits.append((idx, w))
            start = idx + len(w)
    return hits


def _is_negated(text: str, idx: int) -> bool:
    """判断 idx 处的词前面 1~2 个字符是否出现否定词。"""
    prefix = text[max(0, idx - 2):idx]
    return any(neg in prefix for neg in NEGATIONS)


def sense(text: str) -> dict:
    """估计文本的情绪状态，返回 {valence, arousal, intensity, label}。"""
    text = (text or "").strip()
    if not text:
        return {"valence": 0.0, "arousal": 0.0, "intensity": 0.0, "label": "neutral"}

    score = 0.0
    for idx, word in _positions(text, POSITIVE_WORDS):
        score += -1.0 if _is_negated(text, idx) else 1.0
    for idx, word in _positions(text, NEGATIVE_WORDS):
        score += 1.0 if _is_negated(text, idx) else -1.0

    valence = math.tanh(0.7 * score) if score else 0.0

    emoji_pos = sum(1 for ch in text if ch in POSITIVE_EMOJI)
    emoji_neg = sum(1 for ch in text if ch in NEGATIVE_EMOJI)
    if emoji_pos or emoji_neg:
        valence = max(-1.0, min(1.0, valence + 0.25 * (emoji_pos - emoji_neg)))

    intens = sum(1 for w in INTENSIFIERS if w in text)
    punct = len(_AROUSAL_PUNCT.findall(text))
    laughs = len(_LAUGH.findall(text))
    arousal = min(
        1.0,
        0.10 * punct
        + 0.12 * intens
        + 0.15 * (emoji_pos + emoji_neg)
        + 0.05 * laughs
        + 0.04 * min(len(text), 120) / 12,
    )

    intensity = min(1.0, abs(valence) * 0.75 + arousal * 0.45)
    if valence > 0.15:
        label = "positive"
    elif valence < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return {
        "valence": round(valence, 4),
        "arousal": round(arousal, 4),
        "intensity": round(intensity, 4),
        "label": label,
    }


def label_cn(valence: float, arousal: float = 0.0) -> str:
    """把情绪数值翻译成给 LLM 看的中文标签。"""
    if valence > 0.35:
        base = "积极"
    elif valence > 0.15:
        base = "略积极"
    elif valence < -0.35:
        base = "消极"
    elif valence < -0.15:
        base = "略消极"
    else:
        base = "中性"
    if arousal > 0.55:
        base += "/激动"
    elif arousal < 0.15 and abs(valence) < 0.15:
        base += "/平静"
    return base
