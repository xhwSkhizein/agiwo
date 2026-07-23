# Artifact 只索引 Session 工作目录中的文件型产出

Artifact 是 agent 工作目录内独立文件的索引与按需读取句柄，不是所有执行结果的统称。普通文本 report、故障说明、聊天回复与 Decision 一律保持普通文本；只有图片、PDF、日志、额外大文本，以及用户授权外置的超长输入，才登记为 Artifact。文件落在 `{agent_workspace}/sessions/<session_id>/artifacts/`；内存对象包含 `artifact_id`、相对 path、短 summary，以及可选的小体积 inline content（超过大小阈值只保留 path，需要时按 path 读取）。

## Status

accepted

## Considered Options

- 把收尾 report、故障说明和文件引用都叫 Artifact：交接字段统一，但把文本结果与文件存储揉进同一抽象，导致落盘、交付和归档语义膨胀。
- Artifact 只表示文件型产出与外置输入，文本结果保持普通字符串：对象更少，目录语义清晰，agent 可按需读取；需要同时改早期把 report 称作 Artifact 的表述。
- 使用独立 blob 存储服务：适合超大对象，但第一版增加部署面；Session 工作目录已足够承载文件并随归档保留。

## Consequences

- 本 ADR 部分取代 ADR 0005、0006、0007、0012、0015、0016、0017、0020、0021、0023、0024 中把普通文本 report 建模或称为 `report Artifact` 的表述。
- Artifact 不含路由字段；Decision 与文本 report 继续分别承载执行意图与说明性结果。
- AssignmentOutcome 与 ObjectiveDelivered 以普通文本 report 为主内容，ArtifactRefs 只列出附带文件。
- 系统生成的故障 / 重试耗尽 / outcome_unknown / max_steps 收口说明都是普通文本 report，可引用已有文件 Artifact，但不把自己建成 Artifact。
- 超长用户输入外置时：原文物化到 `sessions/<session_id>/artifacts/`，ObjectiveLog 仍保留原始 ObjectiveUserInput；模型上下文在原位置只放 path + summary，agent 通过既有读文件能力按需读取。
- Session 归档必须保留该 Session 的 artifacts 目录；不得只保留 ObjectiveLog 引用而删除文件。
- Console 最终交付视图主区渲染文本 report，文件 Artifact 作为附件列表；时间线可展示 Artifact 登记节点。
