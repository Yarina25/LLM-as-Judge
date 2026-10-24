# LLM-as-Judge 自动化评测

基于大模型裁判（LLM-as-a-Judge）的开放问答自动化评测工具。使用一个配置好的大模型作为裁判，按统一的评分维度与评分标准，对被测模型的回答逐条打分，并自动汇总统计。

适用于主观题评测、双模型对比、回归测试、指令遵循检查等无标准答案的开放问答场景。

## 特性

- **OpenAI 兼容协议**：支持 DeepSeek / 通义千问 / 智谱 GLM / OpenAI / 火山方舟 / Ollama 本地等
- **环境变量与 OpenCompass 评测环境一致**：`LLM_API_KEY` / `LLM_API_BASE` / `LLM_MODEL_NAME`
- **多维度打分**：相关性、正确性、完整性、清晰性（各 0-10 分），加权总分
- **输入灵活**：JSONL（推荐）或 CSV，字段为 `question` / `reference_answer` / `answer`
- **输出完整**：逐条结果 CSV + 汇总报告 JSON + 汇总 Markdown
- **工程化**：并发打分、失败重试、JSON 解析容错

## 目录结构

| 文件 | 作用 |
|---|---|
| [llm_judge_eval.py](llm_judge_eval.py) | 可执行评测脚本，内嵌打分 Prompt，含并发、重试与汇总 |
| [samples_demo.jsonl](samples_demo.jsonl) | 示例评测样本（2 条），可直接用于体验流程 |
| [dryrun_test.py](dryrun_test.py) | 不调用 API 的逻辑验证脚本，用于自检解析与汇总逻辑 |
| [README.md](README.md) | 本说明文档 |

## 快速开始

### 1. 安装依赖

需要 Python 3.9 及以上版本，仅依赖 openai 官方 SDK：

```bash
pip install openai
# 或
python -m pip install openai
```

### 2. 配置环境变量

脚本优先读取环境变量，也可通过命令行参数传入（命令行参数优先级更高）。

**Linux / macOS：**

```bash
export LLM_API_KEY=sk-xxxx
export LLM_API_BASE=https://api.deepseek.com
export LLM_MODEL_NAME=deepseek-chat
```

**Windows PowerShell：**

```powershell
$env:LLM_API_KEY="sk-xxxx"
$env:LLM_API_BASE="https://api.deepseek.com"
$env:LLM_MODEL_NAME="deepseek-chat"
```

### 3. 运行评测

```bash
python llm_judge_eval.py --input samples_demo.jsonl --output results/
```

也可全部通过命令行参数指定：

```bash
python llm_judge_eval.py --input samples.jsonl --output results/ \
    --api-key sk-xxx --base-url https://api.deepseek.com \
    --model deepseek-chat --workers 4
```

### 4. 快速验证（dry-run）

不调用任何 API，仅验证脚本自身的加载、解析与汇总逻辑是否正常：

```bash
python dryrun_test.py
# 全部输出 [OK] 即表示脚本逻辑正常
```

## 样本格式

### JSONL（推荐）

每行一个 JSON 对象，共三个核心字段：

```json
{"question": "什么是大模型评测中的 Pass@1？", "reference_answer": "Pass@1 指模型在单次生成中即给出正确答案的概率。", "answer": "Pass@1 就是模型一次生成就答对的概率。"}
{"question": "OpenCompass 支持哪些数据集？", "reference_answer": "支持 MMLU、GSM8K、C-Eval 等。", "answer": "支持 MMLU 和 GSM8K 等很多数据集。"}
```

### CSV

表头必须包含 `question` 列，建议同时包含 `reference_answer` 与 `answer` 列；其余列作为附加信息保留在样本 meta 中。

### 字段说明

| 字段 | 是否必填 | 说明 |
|---|---|---|
| `question` | 必填 | 待评测的问题 |
| `reference_answer` | 建议 | 参考答案，供裁判评分参照，缺失时提示未提供 |
| `answer` | 建议 | 被测模型的回答，缺失时提示模型未输出 |

## 打分 Prompt

打分从四个维度评估回答质量，每项 0-10 分：

| 维度 | 权重 | 考察内容 |
|---|---|---|
| relevance（相关性） | 0.35 | 是否直接回应问题，是否答非所问或偏离主题 |
| correctness（正确性） | 0.35 | 事实、推理与结论是否正确，是否存在明显错误 |
| completeness（完整性） | 0.20 | 是否覆盖问题全部关键点，是否有重大遗漏 |
| clarity（清晰性） | 0.10 | 结构是否清晰、表达是否通顺、是否易于理解 |

**总分公式：**

```
total = round(0.35 × relevance + 0.35 × correctness + 0.20 × completeness + 0.10 × clarity, 1)
```

总分范围 0-10，保留 1 位小数。Prompt 要求裁判模型只输出结构化 JSON，便于脚本自动解析；同时内置评分标准锚点，尽量降低不同裁判模型之间的评分漂移。完整 Prompt 全文（系统提示词 + 用户模板）内嵌于 [llm_judge_eval.py](llm_judge_eval.py) 中。

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | 必填 | 评测样本文件路径（.jsonl 或 .csv） |
| `--output` | `results` | 输出目录 |
| `--api-key` | 环境变量 | API Key（`LLM_API_KEY`） |
| `--base-url` | 环境变量 | API Base URL（`LLM_API_BASE`） |
| `--model` | 环境变量 | 裁判模型名（`LLM_MODEL_NAME`） |
| `--workers` | `4` | 并发线程数 |
| `--max-samples` | `0` | 最多评测样本数（0 表示全部） |
| `--timeout` | `60.0` | 单次请求超时（秒） |
| `--max-retries` | `3` | 失败重试次数 |

## 输出结果

运行完成后，输出目录下生成三个文件：

- **judge_results.csv**：逐条结果，每个样本一行，包含 question、reference_answer、answer、total、四个维度分、reason（评分理由）、ok（是否成功）、error（失败原因）
- **summary.json**：结构化汇总数据
- **summary.md**：人类可读的汇总报告

### 汇总指标

| 指标 | 含义 |
|---|---|
| judged_ok / judged_fail | 评测成功 / 失败的样本数 |
| score | 全部成功样本的平均总分（0-10） |
| dims | 四个评分维度的均值、最低、最高 |
| distribution | 总分五档分布（0-2 / 2-4 / 4-6 / 6-8 / 8-10） |
| best/worst_sample | 得分最高与最低的样本，便于人工复核 |

## 并发与重试

- 脚本使用线程池并发调用裁判模型，默认 4 个并发线程，可通过 `--workers` 调整
- 单次调用失败后按 1s、2s、4s 指数退避重试，最多 `--max-retries` 次（默认 3 次）
- 若仍失败，该样本标记为失败并继续后续样本，最终在汇总中单独统计

## 常见问题与注意事项

| 问题 / 注意点 | 处理方式 |
|---|---|
| 裁判模型的选择 | 建议使用推理能力较强的模型作为裁判；对同一批样本保持裁判模型不变，否则结果不可比 |
| 评分稳定性 | 脚本已固定 `temperature=0` 降低随机性；如仍不稳定可对同一批样本多次评测取均值 |
| 参考答案缺失 | `reference_answer` 缺失不影响运行，但会降低裁判评分准确性，建议尽量提供 |
| API 限流 | 若触发 429 限流，调低 `--workers` 并适当调大 `--timeout` |
| 密钥安全 | 不要把真实 API Key 写入样本文件或提交到代码仓库 |
| 输出目录 | 重复运行会覆盖同名输出文件，建议按批次使用不同 `--output` 目录 |

## License

本项目仅用于评测学习与研究用途，使用前请遵守所调用模型服务商的条款与当地法律法规。
