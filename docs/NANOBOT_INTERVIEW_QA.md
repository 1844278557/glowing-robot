# nanobot 项目面试回答文档

## 1. 项目整体

### Q1：这个项目解决什么问题？

nanobot 解决的是普通用户低成本部署个人 AI 助手的问题。它不是只做聊天问答，而是把大模型和工具调用、多聊天平台、定时任务、长期记忆、RAG 知识库结合起来，让助手可以在微信、Slack、命令行等渠道里长期运行并执行实际任务。

### Q2：为什么强调轻量化？

很多 Agent 框架功能很全，但代码量大、依赖重、部署和二次开发成本高。nanobot 把核心能力拆成 AgentLoop、AgentRunner、MessageBus、ToolRegistry、ChannelManager 等清晰模块，在保持工具调用和多渠道能力的同时，降低理解和扩展成本。

### Q3：Agent 的核心执行流程是什么？

核心流程是：接收消息 -> 读取会话历史和记忆 -> 构建上下文 -> 调用大模型 -> 判断是否需要工具 -> 执行工具 -> 把工具结果写回上下文 -> 继续推理或输出最终回复。这个循环由 AgentRunner 执行，业务层的会话、渠道、记忆和工具注册由 AgentLoop 管理。

### Q4：AgentLoop 和 AgentRunner 为什么拆开？

AgentLoop 负责产品层逻辑，比如消息总线、会话、记忆、MCP、RAG、Cron 和渠道上下文。AgentRunner 只负责模型和工具的循环执行。这样执行逻辑更纯粹，后续子代理、主代理都可以复用 AgentRunner。

## 2. 上下文设计

### Q5：Agent 的上下文怎么设计？

上下文由 ContextBuilder 分层构建。第一层是 system prompt，包含身份、运行环境、工作区、工具规则和安全规则；第二层读取工作区的 AGENTS.md、SOUL.md、USER.md、TOOLS.md；第三层注入长期记忆 MEMORY.md；第四层注入 Skill 摘要；第五层加入当前会话历史和当前用户消息。

当前时间、渠道和 chat_id 会作为 Runtime Context 放在用户消息前，但明确标记为 metadata，不当作高优先级指令。这样模型知道自己在哪个渠道、当前时间是什么，但不会把这些信息误当系统指令。

### Q6：如何控制上下文过长？

项目会估算当前 prompt token 数。当上下文接近预算上限时，MemoryConsolidator 会选择较早的消息片段，在用户轮次边界处压缩成长期记忆和 Episode，避免切断工具调用链。工具结果也会做 16K 字符截断，防止单次工具输出撑爆上下文。

### Q7：Skill 是怎么进入上下文的？

Skill 不会一开始全部展开。系统先扫描 workspace/skills 和内置 skills，生成包含名称、描述、路径和可用状态的 skills summary。模型需要某个技能时，再通过 read_file 读取对应 SKILL.md。这是渐进式加载，减少无关上下文占用。

## 3. 消息网关与多渠道

### Q8：项目里的 gateway 是什么？

gateway 是 nanobot 的常驻运行入口，不是传统 HTTP API Gateway。它负责加载配置、创建 Provider、MessageBus、AgentLoop、ChannelManager、CronService 和 HeartbeatService，并用 asyncio.gather 同时运行 Agent 和所有聊天渠道。

### Q9：多平台消息怎么统一？

每个平台只负责把原始消息转换成统一的 InboundMessage，字段包括 channel、sender_id、chat_id、content、media、metadata。Agent 处理后生成统一的 OutboundMessage，再由 ChannelManager 根据 channel 找到对应 Channel 发送回平台。

### Q10：如果用户同时在微信和 Slack 发消息，网关怎么处理？

微信和 Slack 都会把消息放入同一个 MessageBus.inbound。每条消息的 session_key 是 channel:chat_id，比如 weixin:user_001、slack:channel_abc。同一个 session 用锁串行处理，避免上下文写乱；不同 session 可以并发处理，但受全局并发信号量控制，默认最多 3 路。

### Q11：怎么接入新的聊天平台？

新增一个继承 BaseChannel 的适配器，实现 start、stop、send，必要时实现 send_delta 和 login。接收方向把平台消息转成 InboundMessage，发送方向把 OutboundMessage 转回平台消息。内置渠道通过 pkgutil 扫描，外部插件通过 entry_points 加载。

### Q12：pkgutil 和 entry_points 的作用是什么？

pkgutil 用来扫描 nanobot/channels 目录里的内置渠道文件，比如 weixin.py、slack.py。entry_points 用来加载第三方安装包提供的外部渠道插件。这样新增渠道不需要修改核心代码，系统启动时能自动发现。

### Q13：ChannelManager 是什么？

ChannelManager 是聊天渠道总管。它读取配置，初始化已启用的 Channel，启动和停止渠道，从 MessageBus.outbound 消费回复消息，并根据 channel 字段路由到对应平台。它还负责流式消息合并、发送失败重试和 allowFrom 基础校验。

## 4. 微信渠道

### Q14：微信渠道具体怎么做？

微信渠道是 BaseChannel 的一个适配器，使用 HTTP long-poll 方案。启动前通过二维码登录获取 bot_token，并把 token、get_updates_buf、context_token 等状态保存到本地 account.json。重启时优先读取本地状态，减少重复扫码。

接收消息时，WeixinChannel 调用 getupdates 拉取新消息，用 get_updates_buf 做游标，用 message_id 或 seq 去重，过滤机器人自己发出的消息。然后解析 item_list，支持文本、引用、图片、语音、文件、视频等类型。媒体会下载到本地，语音优先使用微信返回的识别文本，最后封装为 InboundMessage 投递给 MessageBus。

发送消息时，Agent 生成 OutboundMessage，ChannelManager 找到 WeixinChannel。文本会按微信单条长度限制拆分发送；文件、图片、视频会先上传到微信 CDN，再发送媒体消息。微信回复依赖 context_token，所以接收消息时会缓存每个用户的 context_token。

### Q15：微信为什么要做 context_token 缓存？

微信接口发送回复时需要对应用户的 context_token。它来自用户发来的消息。如果没有缓存，系统就不知道该如何在微信上下文中回复这个用户，所以 WeixinChannel 在接收消息时会按 from_user_id 保存 context_token。

## 5. 发送消息与流式输出

### Q16：Agent 怎么发送消息？

普通回复路径是：Channel 收消息 -> InboundMessage -> MessageBus.inbound -> AgentLoop -> 模型和工具执行 -> final_content -> OutboundMessage -> MessageBus.outbound -> ChannelManager -> 对应 Channel.send。

如果 Agent 要主动发文件或中途发消息，会调用 message 工具。AgentLoop 每轮会把当前 channel、chat_id、message_id 设置到 message 工具里，所以模型可以直接调用 message(content=..., media=[...])，MessageTool 再把消息投递到 outbound 队列。

### Q17：流式输出怎么处理？

支持流式的 Channel 会在消息 metadata 中加入 _wants_stream。AgentLoop 收到后注册 on_stream 回调，把模型生成的 delta 包装成带 _stream_delta 和 _stream_id 的 OutboundMessage。结束时发送 _stream_end。ChannelManager 会合并连续 delta，减少 IM 平台 API 调用，再调用 send_delta 或退化成最终完整回复。

### Q18：发送失败怎么办？

ChannelManager 统一做发送重试。如果 Channel.send 或 send_delta 抛异常，会按 1s、2s、4s 指数退避重试。最终失败会记录日志，但不会让整个 Agent 进程崩掉。

## 6. 工具、MCP 与 Skill

### Q19：ReAct 工具调用怎么做？

模型根据上下文决定是否调用工具。ToolRegistry 负责注册工具、暴露 schema、校验参数、类型转换和执行工具。工具结果会作为 tool message 写回上下文，模型再根据结果继续推理或输出最终答案。

### Q20：项目有哪些工具？

内置工具包括文件读写、文件编辑、目录浏览、Shell 执行、Web 搜索、网页抓取、消息发送、定时任务、子代理、Episode 记忆检索、RAG 文档索引和检索等。

### Q21：为什么支持 MCP？

MCP 用来扩展外部工具和数据源。项目支持 stdio、SSE、streamableHttp 三种传输方式，会把 MCP server 暴露的工具 schema 转成 nanobot 原生 Tool，注册到 ToolRegistry。这样内置工具和外部 MCP 工具可以走同一套调用流程。

### Q22：Skill 具体怎么写？有什么经验？

Skill 是一个 SKILL.md，写给 Agent 看的任务说明。好的 Skill 应包含适用场景、输入要求、执行步骤、工具使用、输出格式、失败处理和依赖声明。经验是不要写百科式文档，要写成可执行流程；依赖脚本尽量放在 skill 目录里，让 Agent 调脚本而不是临时拼复杂命令。

## 7. 子代理

### Q23：如何实现子代理机制？

主 Agent 通过 spawn 工具启动子代理。SubagentManager 生成 task id，用 asyncio.create_task 启动后台任务。子代理复用 AgentRunner，但有独立 system prompt 和独立工具集，主要包含文件、Shell、Web 搜索等工具，不包含 message 和 spawn，避免子代理直接发消息或递归创建子代理。

子代理完成后不会直接回复用户，而是把结果包装成 channel=system、sender_id=subagent 的内部消息投递回 MessageBus，由主 Agent 再总结成自然语言发给用户。

### Q24：什么时候适合用子代理？

适合复杂、耗时、可独立完成的任务，比如后台资料搜索、代码分析、长任务执行。主 Agent 可以继续保持响应，子代理完成后再把结果交回主 Agent。

## 8. Cron 与 Heartbeat

### Q25：Cron 服务怎么做？

Cron 负责明确时间点或周期任务。CronTool 让模型可以 add、list、remove 任务。任务支持 at、every、cron 三种形式，并支持时区。CronService 把任务持久化到 workspace/cron/jobs.json，保存下次执行时间、上次状态、错误和最近 20 次运行历史。服务只为最近一次任务挂 timer，到点后执行 due jobs，再计算下一次唤醒。

### Q26：Heartbeat 服务怎么做？

Heartbeat 负责周期性主动检查任务。它定期读取 HEARTBEAT.md，然后让 LLM 通过虚拟 heartbeat 工具返回 skip 或 run。如果返回 run，就把任务交给完整 Agent 流程执行。执行完后再调用 evaluate_response 判断是否值得通知用户，避免无变化的例行检查频繁打扰用户。

### Q27：Cron 和 Heartbeat 的区别是什么？

Cron 是确定性调度，比如明天 9 点提醒、每小时检查一次。Heartbeat 是周期性让 Agent 自己判断 HEARTBEAT.md 里有没有需要主动执行的任务。Cron 更像闹钟，Heartbeat 更像主动巡检。

## 9. 记忆与 RAG

### Q28：长期记忆怎么做？

项目使用 MEMORY.md 和 Episode 双层记忆。MEMORY.md 保存长期事实，Episode 保存结构化历史对话片段。触发压缩时，模型通过 save_memory 工具提取 topic、summary、key_points、decisions、entities、importance、tags，并更新长期记忆。

### Q29：记忆压缩失败怎么办？

如果模型没有按要求调用 save_memory，系统会降级处理。连续失败 3 次后会 raw archive，把原始消息归档成 Episode，保证历史信息不丢失。

### Q30：RAG 怎么做？

RAG 用 ChromaDB 保存本地文档向量。文档先解析成文本，再根据类型选择分块策略：Markdown 按标题，Python 代码按 AST 类/函数边界，普通文本用语义分块或滑动窗口。检索时结合向量检索、BM25、RRF 融合排序和可选 rerank，提高本地资料问答准确性。

### Q31：RAG 的数据结构怎么设计？

RAG 数据结构分成“文档级元数据”和“分块级向量数据”两层。文档级用 `document_registry.json` 记录每个文件的路径、类型、大小、修改时间、chunk 数、索引时间、过期时间和状态，用于去重、增量更新和 TTL 清理。分块级用 ChromaDB 的 `rag_documents` collection 存储，每个 chunk 有唯一 `chunk_id`、正文内容、向量和 metadata，metadata 包含 `file_path`、`file_name`、`chunk_index`、`doc_type` 以及标题、代码行号等结构信息。

检索时，向量检索直接从 ChromaDB 返回相关 chunk；BM25 会基于 ChromaDB 中的文档块懒加载构建关键词索引；最后用 `source_file + chunk_index` 作为唯一键做 RRF 融合，返回带来源、分数和 metadata 的结果给模型。

### Q32：记忆和 RAG 的区别是什么？

记忆面向历史对话和用户偏好，解决“之前聊过什么、用户有什么长期信息”的问题。RAG 面向外部或本地文档，解决“基于资料回答问题”的问题。

## 10. 调试、性能、安全与评估

### Q33：如何调试 Agent 异常行为？

按链路排查：先用 CLI 和日志复现，再看 AgentRunner 的 stop_reason、tools_used、tool_events；然后检查上下文是否过长或指令冲突；再看会话 JSONL、MEMORY.md、Episode、RAG 索引；如果是平台问题，就检查 InboundMessage 和 OutboundMessage 是否正确。

### Q34：如何优化 Agent 响应速度？

主要从四方面优化：上下文压缩，减少无关历史；工具并发和 timeout，避免慢工具阻塞；模型选择和 max_tokens 控制，降低推理成本；通道层做流式输出和 delta 合并，让用户更早看到回复，同时减少平台 API 调用。

### Q35：需要考虑哪些安全问题？

重点是工具误用、文件越权、Shell 风险、SSRF、恶意网页内容注入、MCP 插件风险和渠道滥用。项目通过 allowFrom、restrictToWorkspace、禁用 Shell 配置、URL 内网地址拦截、网页不可信标记、HTML 清洗、工具参数校验和结果截断来降低风险。

### Q36：如何防止 Prompt 注入通过网关？

网关主要做消息路由，不直接理解内容。防护要分层做：渠道 allowFrom 控制来源；外部网页内容标记为不可信；工具按 schema 校验参数；文件和 Shell 限制在工作区；Web 抓取做 SSRF 拦截；生产环境还可以加敏感指令过滤、频控和高危工具二次确认。

### Q37：如何评估 Agent 性能和质量？

性能指标看端到端延迟、首 token 时间、模型耗时、工具耗时、并发成功率、消息发送失败率、Token 消耗和内存占用。质量指标看任务完成率、工具选择准确率、RAG 命中率、回答准确性、幻觉率、安全边界遵守情况和用户是否需要反复纠正。

评估方式是准备固定测试集，覆盖文件操作、Web 搜索、RAG 问答、定时任务、多轮记忆、异常工具调用和多渠道消息。每次改动后用同一批任务对比完成率、耗时和错误类型。

## 11. 可靠性与架构取舍

### Q38：网关挂了怎么办？

当前 gateway 是轻量单进程，挂了以后渠道、Agent、Cron、Heartbeat 都会停止。项目通过落盘会话、记忆、Cron job、RAG 索引、微信 token 和游标来支持重启恢复。生产部署需要 systemd、Docker restart policy 或 Kubernetes 做进程守护。

### Q39：为什么不用 Kafka 做网关？

项目定位是个人 AI 助手，主要是低并发、轻量单机部署。用 asyncio.Queue 可以减少依赖和部署成本。Kafka 更适合高吞吐、多消费者、分布式持久化场景。如果未来要多实例高可用，可以把 MessageBus 抽象成接口，替换成 Redis Stream、RabbitMQ 或 Kafka。

### Q40：这个项目最难的点是什么？

难点不是单个模块，而是把模型、工具、记忆、RAG、多渠道、安全和异步任务串成稳定闭环。尤其要处理上下文污染、工具失败、渠道重试、并发会话、长期状态持久化这些边界问题。

### Q41：如果继续优化，你会做什么？

优先做三件事：第一，增加工具权限分级和高危操作确认；第二，建设固定评估集和链路观测，记录每轮工具、耗时、Token 和失败原因；第三，把 MessageBus 做成可插拔实现，为未来 Redis/Kafka 多实例部署留扩展口。

## 12. 详答补充版

这一部分用于回答面试官追问“能不能讲细一点”。前面的问题适合快速回答，下面内容适合展开说明。

### Agent 上下文怎么设计？

可以从“五层上下文”来讲。

第一层是基础系统身份。`ContextBuilder` 会生成 system prompt，告诉模型自己是 nanobot，当前运行环境、Python 版本、工作区路径是什么，并写入行为规范，例如修改文件前先读文件、工具失败后先分析错误、网页内容是不可信数据、发送文件必须调用 `message` 工具。

第二层是工作区启动文件。项目会读取 `AGENTS.md`、`SOUL.md`、`USER.md`、`TOOLS.md`。这些文件让同一个 Agent 在不同工作区可以有不同规则，比如项目规范、用户偏好、工具说明。

第三层是长期记忆。`memory/MEMORY.md` 里的长期事实会作为 `# Memory` 注入上下文，用来保存用户偏好、长期项目背景和重要决定。

第四层是 Skill 摘要。系统不会把所有 Skill 全量塞进上下文，而是先放技能名称、描述和路径。真正需要某个技能时，模型再用 `read_file` 读取具体 `SKILL.md`。这样能减少无关上下文。

第五层是会话历史和当前消息。会话按 `channel:chat_id` 隔离，例如 `weixin:user_001` 和 `slack:channel_abc` 是两个独立 session。当前消息前会加入 Runtime Context，包括当前时间、渠道和 Chat ID，但明确标记为 metadata，不让模型把它当成系统指令。

如果面试官继续问“为什么这样设计”，可以回答：这样做能同时满足个性化、记忆、技能扩展和安全隔离；同时上下文来源清晰，调试时能知道问题来自系统规则、用户规则、记忆还是当前消息。

### Agent 怎么发送消息？

Agent 发送消息有两条路径：普通最终回复和 `message` 工具主动发送。

普通最终回复的链路是：

```text
用户在微信/Slack 发消息
-> Channel 转成 InboundMessage
-> MessageBus.inbound
-> AgentLoop 消费消息
-> ContextBuilder 构建上下文
-> AgentRunner 调模型和工具
-> 得到 final_content
-> AgentLoop 包装成 OutboundMessage
-> MessageBus.outbound
-> ChannelManager 分发
-> 对应 Channel.send()
-> 用户收到回复
```

主动发送用于文件、图片、音频、文档等交付场景。每轮处理前，`AgentLoop` 会把当前 `channel`、`chat_id`、`message_id` 设置到 `message` 工具。模型调用 `message(content=..., media=[...])` 时，`MessageTool` 会生成 `OutboundMessage` 放入 outbound 队列。如果本轮已经通过 `message` 工具发过消息，AgentLoop 会避免再次发送最终回复，防止重复。

流式输出则走 `_stream_delta` 和 `_stream_end`。模型生成一个 delta，就包装成一条流式 `OutboundMessage`；工具调用前会结束当前流片段，工具结束后再继续。`ChannelManager` 会合并连续 delta，减少微信、Slack 等平台的 API 压力。

### Gateway 是怎么工作的？

`gateway` 是 nanobot 的常驻运行入口，不是传统 HTTP API Gateway。它启动后创建并连接这些核心对象：

```text
MessageBus
AgentLoop
ChannelManager
CronService
HeartbeatService
Provider
SessionManager
```

核心是内部消息总线。所有聊天渠道收到消息后都转成 `InboundMessage` 放入 `MessageBus.inbound`；Agent 处理后把 `OutboundMessage` 放入 `MessageBus.outbound`；`ChannelManager` 再按 `channel` 字段发回微信、Slack 等平台。

如果微信和 Slack 同时发消息，它们进入同一个 inbound 队列，但 session_key 不同。项目对同一个 session 加锁串行处理，避免上下文写乱；不同 session 可以并发处理，但受默认 3 路全局并发限制。

网关挂了以后，进程内任务会停止，但会话、记忆、Cron job、RAG 索引、微信登录状态都有本地持久化。生产环境需要 systemd、Docker restart policy 或 Kubernetes 做进程守护。

### 微信渠道具体怎么做？

微信渠道是 `BaseChannel` 的一个具体适配器。它使用 HTTP long-poll 方式，不依赖本地微信客户端。

登录阶段：调用微信接口获取二维码，用户扫码确认后获得 `bot_token`，再把 token、消息游标 `get_updates_buf`、用户 `context_token` 保存到本地 `account.json`。重启时优先读取本地状态，减少重复登录。

接收阶段：渠道循环调用 `getupdates` 拉取新消息，带上 `get_updates_buf` 作为游标。收到消息后，先过滤机器人自己发出的消息，再用 `message_id` 或 `seq` 去重。随后解析 `item_list`，支持文本、引用、图片、语音、文件和视频。媒体会下载到本地媒体目录；语音优先使用微信返回的识别文本，没有时再尝试转写。最后封装成 `InboundMessage(channel="weixin")` 投递给 MessageBus。

发送阶段：Agent 生成 `OutboundMessage(channel="weixin")`，ChannelManager 找到 WeixinChannel。微信回复需要 `context_token`，所以接收消息时会按用户缓存。文本会按微信单条长度限制拆分；媒体文件会先上传到微信 CDN，再发送媒体消息。

面试总结句：微信渠道只做协议适配，把微信消息翻译成 nanobot 标准消息，也把 nanobot 回复翻译回微信消息，Agent 核心不感知微信协议。

### 子代理机制怎么实现？

主 Agent 通过 `spawn` 工具创建子代理。`SubagentManager` 生成 task id 后，用 `asyncio.create_task()` 启动后台任务。

子代理复用 `AgentRunner`，但有独立 system prompt 和独立工具集。它能读写文件、执行 Shell、搜索网页，但没有 `message` 和 `spawn` 工具。这样可以避免子代理直接发消息给用户，也避免子代理递归创建更多子代理。

子代理完成后不会直接回复用户，而是把结果包装成 `channel="system"`、`sender_id="subagent"` 的内部消息放回 MessageBus。主 Agent 收到这条内部消息后，再用自然语言总结给用户。

这个设计的好处是：耗时任务可以后台跑，同时由主 Agent 统一对外表达，用户体验更一致。

### Cron 和 Heartbeat 怎么做？

Cron 处理确定性时间任务。模型通过 `cron` 工具添加任务，支持 `at`、`every` 和 `cron_expr` 三种方式。任务会保存到 `workspace/cron/jobs.json`，包括任务 id、调度规则、下次执行时间、上次状态、错误和运行历史。CronService 启动后只计算最近一次唤醒时间，到点执行 due jobs，再重新计算下一次唤醒。

Heartbeat 处理主动巡检。它定期读取 `HEARTBEAT.md`，然后让 LLM 通过虚拟 `heartbeat` 工具返回 `skip` 或 `run`。如果是 `run`，就把任务交给完整 Agent 流程处理。执行结束后，再用 `evaluate_response()` 判断是否值得通知用户，避免无意义的例行检查频繁打扰。

区别是：Cron 像闹钟，适合明确时间；Heartbeat 像巡检，适合周期性判断有没有该做的事。

### RAG 的数据结构怎么设计？

可以按“文档表、chunk 表、检索结果对象”三层来讲。

第一层是文档级注册表。项目会在 `memory/rag/document_registry.json` 里维护已索引文件的元信息，key 是 `file_path`，value 是 `DocumentRecord`。里面保存文件名、文档类型、chunk 数、文件修改时间、文件大小、索引时间、过期时间和状态。它的作用是判断文件有没有变更、是否需要重新索引，以及做 TTL 过期清理。

第二层是分块级向量数据。解析后的文档会被切成 `DocumentChunk`，每个 chunk 有 `chunk_id`、`file_path`、`content`、`chunk_index`、`doc_type` 和 metadata。写入 ChromaDB 时，`content` 作为 document，embedding 由 ChromaDB 的 embedding function 生成，metadata 保存来源文件、文件名、chunk 序号、文档类型，以及标题路径、代码函数名、起止行号等结构信息。`chunk_id` 由文件路径 hash 和 chunk 序号组成，方便更新和删除指定文件的所有分块。

第三层是检索结果结构。向量检索会返回 chunk id、文本、metadata 和 distance；BM25 会从 ChromaDB 取出所有 chunk 文本后懒加载构建关键词索引。最终用 `source_file + chunk_index` 作为唯一键做 RRF 融合，封装成 `RetrievalResult`，包含正文、来源文件、chunk 序号、相关分数和 metadata，再由 `RAGSearchResult` 格式化进模型上下文。

面试总结句：RAG 不是只存一段文本向量，而是文档级注册表负责生命周期管理，chunk 级向量库负责语义检索，metadata 负责来源追踪和过滤，检索结果对象负责把命中的片段稳定地交给模型。

### Skill 怎么写，经验是什么？

Skill 是一个 `SKILL.md`，它不是可执行代码，而是给 Agent 的任务操作手册。项目会扫描内置 Skill 和工作区 Skill，先把摘要放进上下文，需要时再读取全文。

一个好的 Skill 应该包含：

- 适用场景：什么时候触发。
- 输入要求：需要用户提供什么。
- 执行步骤：按顺序怎么做。
- 工具使用：该用哪些工具。
- 禁止事项：哪些情况不要做。
- 输出格式：结果怎么返回。
- 失败处理：依赖缺失、权限不足、结果为空时怎么办。

经验是：Skill 要写得像 SOP，不要写成百科。复杂逻辑尽量沉到脚本里，Skill 负责告诉 Agent 什么时候调用脚本、如何解释结果。

### 如何调试 Agent 异常行为？

排查顺序是：输入、上下文、模型、工具、状态、渠道。

先用 CLI 复现并开启日志，看模型是否返回错误、是否进入工具调用。再看 `stop_reason`、`tools_used`、`tool_events`，判断是工具选错、参数错、工具失败还是达到最大迭代。然后检查上下文，包括 system prompt、bootstrap 文件、memory、skills summary 和 session history。

如果问题和历史有关，看会话 JSONL、MEMORY.md、Episode。若是 RAG 问答异常，看文档是否索引、chunk 是否合理、检索结果是否相关。若是平台收发异常，看 `InboundMessage` 和 `OutboundMessage` 是否正确。

核心原则：不要直接认为“模型不行”，要把 Agent 行为拆成上下文、模型、工具、状态和渠道逐段定位。

### 如何优化 Agent 响应速度？

响应速度可以从五个方向优化。

第一是上下文压缩。减少无关历史，通过记忆压缩把旧消息转成 MEMORY 和 Episode。  
第二是工具执行。无依赖工具并发执行，工具设置 timeout，失败快速返回。  
第三是模型选择。普通任务用更快模型，复杂任务再用强模型；控制 `max_tokens` 和 reasoning 配置。  
第四是 RAG 控制。控制 `top_k`，只在高精度场景启用 rerank。  
第五是通道体验。流式输出让用户更早看到内容，ChannelManager 合并 delta 减少平台 API 调用。

面试可以总结：优化不是只看最终耗时，还要看首 token 时间、工具耗时、模型耗时和渠道发送耗时。

### 需要考虑哪些安全问题？

主要风险包括：

- 渠道被陌生人调用。
- 文件越权读取或写入。
- Shell 工具执行危险命令。
- SSRF 访问内网。
- 网页内容或 RAG 文档 Prompt 注入。
- MCP 插件不可信。
- 工具结果过大污染上下文。

项目里的解决方式包括：

- `allowFrom` 控制谁能访问渠道。
- `restrictToWorkspace` 限制文件和 Shell 在工作区内。
- Shell 工具可配置关闭。
- WebFetch 做 URL 协议、域名和 IP 校验，拦截内网地址。
- 外部网页内容标记为不可信数据。
- HTML 清洗。
- 工具参数按 schema 校验。
- 工具结果 16K 截断。

如果继续增强，会做工具权限分级、高危操作二次确认、MCP 白名单、审计日志和频控。

### 如何评估 Agent 性能和质量？

性能指标包括：端到端延迟、首 token 时间、模型调用耗时、工具耗时、消息发送失败率、并发成功率、Token 消耗和内存占用。

质量指标包括：任务完成率、工具选择准确率、工具参数正确率、RAG 命中率、回答准确性、幻觉率、安全边界遵守情况和用户纠正次数。

评估方法是准备固定测试集，覆盖文件操作、Web 搜索、RAG 问答、定时任务、多轮记忆、异常工具调用和多渠道消息。每次改动后用同一批任务回归，对比完成率、耗时和错误类型。

项目里已有一些基础抓手，例如 AgentRunner 会记录 `usage`、`tools_used`、`tool_events` 和 `stop_reason`。后续可以在这些基础上做更完整的评估报表。
