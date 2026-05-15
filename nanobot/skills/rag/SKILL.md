---
name: rag
description: Index documents into RAG knowledge base and retrieve relevant content for Q&A. Unlike summarize (one-time extraction), RAG enables continuous Q&A on indexed documents.
metadata: {"nanobot":{"emoji":"📚","requires":{"pip":["chromadb","markitdown"]},"optional_pip":["python-docx","rank-bm25","jieba","sentence-transformers","numpy"]}}
---

# RAG - 文档知识库检索

将用户上传的文档（PDF、DOCX、Markdown、代码等）索引到向量知识库，
支持基于文档内容的持续问答。

## 与 summarize 的区别

| 特性 | RAG | summarize |
|------|-----|-----------|
| 用途 | 持续性问答 | 一次性摘要/提取 |
| 存储 | 索入向量库，可反复查询 | 即用即弃 |
| 触发 | 用户上传文档 + 提问具体内容 | "总结这个文件/链接" |

## When to use (触发条件)

**同时满足以下条件时使用本技能：**
1. 用户消息中包含文件路径（如 `[file: /path/to/doc.pdf]`）
2. 用户针对该文档提出了**具体的查询问题**（而非简单的"总结一下"）

**典型触发示例：**
- "这个PDF里关于XXX是怎么说的？"
- "帮我查一下这份文档中关于YYY的章节"
- "索引这个文档，然后回答我的问题"
- 用户上传文件后紧跟具体问题

**不应触发的场景：**
- 仅要求总结/概括 → 使用 summarize 技能
- 一般性问题，无文档关联 → 正常对话即可
- 文件是代码但问题是关于运行/调试 → 使用 shell/代码工具

## 可用工具

| 工具名 | 用途 |
|--------|------|
| `index_document` | 索引文档到RAG知识库（自动设置过期时间） |
| `search_rag` | 在知识库中检索相关内容（自动排除过期文档） |
| `remove_rag_document` | 移除指定已索引文档 |
| `list_rag_documents` | 列出所有已索引文档及状态/过期时间 |
| `clear_rag_store` | 清空整个知识库（需确认） |
| `cleanup_expired_rag` | 清理已过期的文档，释放存储空间 |

## 工作流程

### Step 1: 索引文档

当用户上传了新文档时，首先调用 `index_document` 工具：

```
index_document(file_path="/path/to/document.pdf")
```

- 自动检测文件类型（PDF/DOCX/MD/TXT/代码）并选择最优分块策略
- 已索引且未修改的文件会自动跳过；强制重新索引用 `force_reindex=true`
- 文档有默认30天过期时间（可通过配置调整），过期后自动标记为 stale

### Step 2: 检索问答

使用 `search_rag` 工具搜索相关知识：

```
search_rag(query="用户的具体问题", top_k=5)
```

- **混合检索**：BM25关键词 + 向量语义检索，RRF融合排序
- **可选重排序**：配置SiliconFlow API后启用Qwen3-Reranker-8B重排序
- 过期文档不参与检索结果

### Step 3: 管理文档（可选）

```
list_rag_documents()           # 查看所有已索引文档及状态（含过期时间）
remove_rag_document(file_path="/path/to/doc.pdf")  # 移除单个文档
cleanup_expired_rag()          # 清理过期文档，释放空间
clear_rag_store(confirm="yes") # 清空全部（危险操作）
```

### 完整交互示例

```
User: [file: /path/to/report.pdf] 这份报告里提到的Q3营收数据是多少？

Agent: （调用 index_document 索引PDF）
      （调用 search_rag query="Q3营收数据"）
      → 根据检索到的片段回答用户问题
```

## 支持的文档格式

| 格式 | 解析方式 | 分块策略 |
|------|---------|---------|
| PDF | MarkItDown | SemanticSplitter |
| DOCX/DOC | python-docx | SemanticSplitter |
| Markdown | 直接读取 | MarkdownHeaderSplitter |
| Python代码 | AST解析 | CodeASTSplitter |
| 其他代码 | 直接读取 | SlidingWindowSplitter |
| 纯文本/TXT | 直接读取 | SemanticSplitter |

## 嵌入模型说明

系统使用 **ChromaDB 内置默认嵌入函数** 进行向量化存储和语义检索。
当前默认模型为 `all-MiniLM-L6-v2`（sentence-transformers），对中文支持有限。
建议通过配置 `rag.embedding_model` 切换为中文优化模型（如 `BAAI/bge-small-zh-v1.5`）。
