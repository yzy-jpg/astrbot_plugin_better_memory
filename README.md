# 更好的记忆 · Better Memory

> AstrBot 插件：把 AI 的「记得你」拆成一套分层认知系统，而不是每次都在向量库里查资料。

大多数「AI 记忆」插件只做一件事：把对话塞进向量库，按语义相似度检索。结果是 AI 像一个查资料的机器——
能「查到」，但并不「记得」。更自然的方式是模仿人类记忆的分层结构：

| 记忆层 | 对应实现 | 生命周期 |
| --- | --- | --- |
| 工作记忆 | AstrBot 自己的上下文窗口 | 单次会话 |
| 情景记忆 | 每次交互留一条带时间戳、情绪标签的小事件 | 长期，随时间衰减 |
| 语义记忆 | 自动抽取「我叫 / 我喜欢 / 我住在 / 记住…」的事实，带信心值 | 长期，相对稳定 |
| 人格记忆 | 可锁定的关系锚点 + 周期性复盘结论 | 永不衰减，每轮必注入 |

## 核心特性

1. **情感感知检索**
   检索打分不看单一相似度，而是四项加权（权重可在 WebUI 调）：

   score = 0.45 x 语义相似度 + 0.20 x 鲜活度 + 0.20 x 情绪一致性 + 0.15 x 新近度

   难过时更容易想起难过的事，开心时更容易想起开心的事——像人一样。

2. **动态遗忘与再巩固**
   - 有效强度按半衰期（默认 30 天）指数衰减；
   - 被回忆时强度回升（记忆再巩固），访问次数计入鲜活度；
   - 被 /mem pin 锁定的核心记忆不衰减，每轮无条件注入。

3. **主动写入**
   - 规则抽取：识别「我叫… / 我喜欢… / 我讨厌… / 我住在… / 我的生日是… / 记住…」等表达，带信心值入库；
   - 可选 LLM 抽取（默认关闭）：让模型判断这条消息里有没有值得长期记住的事。

4. **周期性复盘**
   每 N 条消息（默认 60）让模型回顾近期记忆，提炼新的人格画像 / 关系规则，写回核心记忆——
   「进化结论」不依赖检索是否命中，而是每轮都在系统提示里生效。

5. **完全本地**
   记忆存在插件数据目录的 SQLite 里，不上传任何第三方服务；只有复盘 / LLM 抽取会调用你自己配置的模型。

## 安装

### 从插件市场（推荐）
AstrBot WebUI → 插件 → 插件市场 → 搜索「更好的记忆」→ 安装。

### 手动安装
把本目录放到 AstrBot 的 data/plugins/astrbot_plugin_better_memory/，重载插件即可。

依赖：无额外依赖（jieba、rank-bm25 已随 AstrBot 安装）。

## 命令

| 命令 | 说明 |
| --- | --- |
| /mem（别名 /记忆） | 记忆概览 + 帮助 |
| /mem core | 查看人格 / 关系锚点（永不遗忘） |
| /mem list [类型] | 列出记忆，类型可选 episodic / semantic / core |
| /mem add 内容 | 手动写入长期记忆（会自动识别「生日」等特殊事实并锁定） |
| /mem search 关键词 | 按情感感知评分检索 |
| /mem pin 编号 / unpin 编号 | 锁定 / 解锁某条记忆 |
| /mem forget 编号 | 删除 |
| /mem reflect | 立即复盘（消耗一次模型调用） |
| /mem export | 导出 JSON + 人类可读的 MEMORY_*.md |
| /mem reset confirm | 清空自己的记忆（不可恢复） |

## 配置（WebUI → 插件 → 更好的记忆）

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| enabled | true | 总开关 |
| inject_enabled | true | 是否把记忆注入 System Prompt |
| episodic_enabled | true | 是否记录情景记忆 |
| auto_write | true | 是否用规则主动写入事实 |
| llm_extract | false | 是否用 LLM 抽取事实（每条多一次模型调用） |
| reflect_enabled / reflect_every | true / 60 | 是否复盘、每多少条消息复盘 |
| max_inject_chars | 1200 | 注入字符预算 |
| top_k_episodic / top_k_semantic | 6 / 5 | 召回条数 |
| half_life_days | 30 | 遗忘半衰期 |
| min_episodic_importance / min_fact_importance | 0.30 / 0.45 | 写入门槛 |
| weights.semantic / vividness / emotional / recency | 0.45 / 0.20 / 0.20 / 0.15 | 检索权重 |
| verbose | false | 调试日志 |

## 数据存放在哪

- SQLite：AstrBot/data/plugin_data/astrbot_plugin_better_memory/memory.sqlite3
- 人类可读备份：同目录 MEMORY_<平台_用户>.md（复盘 / export 时生成）
- 按 平台:发送者ID 隔离 scope，不同用户互不可见

## 工作原理（简版）

1. 收到消息 → 情绪感知（词表 + 否定词 + 表情 + 标点，输出 valence / arousal / intensity）→ 写情景记忆；
2. 正则抽取事实 → 写语义 / 核心记忆；
3. 每 N 条 → 复盘：LLM 回顾近期记忆 → 新核心记忆 / 事实；
4. LLM 请求前 → 候选记忆按四项加权打分 → 取 top-k + 核心记忆 + 复盘 → 拼进 System Prompt → 被召回的记忆强度回升、访问计数 +1。

## 开发与自测

    python -m py_compile main.py memory.py emotions.py   # 语法检查
    python test_memory_core.py                           # 离线单元测试（不需要 AstrBot 运行）

端到端验证可以用假 OneBot 客户端（本机部署时放在 qqbot/tools/demo_cmd2.py），无需登录 QQ 即可验证
「消息进 → 记忆写入 → 下一轮注入 → 模型复述」。

## 已知限制

- 情绪感知基于中文词表 + 启发式，不理解反讽 / 复杂语境（可开 llm_extract 缓解）；
- 语义相似度用 BM25（jieba 分词），不是向量检索；对近义改写不敏感，但零额外依赖、无 embedding 成本；
- 情景记忆按消息条粒度存储，长对话会持续增长（可用 /mem forget 或调低 half_life_days 控制召回）；
- 复盘依赖模型可用，模型不可用时静默跳过。

## Roadmap

- [ ] 可选接入 embedding provider 做向量召回（与 BM25 混合）
- [ ] 群聊 / 私聊不同 scope 策略
- [ ] WebUI 记忆管理页（可视化编辑关系锚点）
- [ ] 记忆导入 / 导出加密

## 许可

AGPL-3.0-or-later（与 AstrBot 生态一致）。详见 LICENSE。

## 致谢

设计参考了分层记忆 / 情感嵌入 / 记忆再巩固等公开研究思路，以及 AstrBot 插件生态的最佳实践。
