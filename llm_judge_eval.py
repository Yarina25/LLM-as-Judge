#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-Judge 自动化评测脚本
============================

用途：对"开放/主观问答"类评测样本，使用一个"裁判模型"(Judge Model) 按
多维评分标准打分，并输出逐条结果与汇总统计。

特性：
- OpenAI 兼容协议：支持 DeepSeek / 通义千问 / 智谱 GLM / OpenAI / 火山方舟 / Ollama 本地等
- 环境变量与 OpenCompass 评测环境保持一致：LLM_API_KEY / LLM_API_BASE / LLM_MODEL_NAME
- 输入：JSONL（推荐）或 CSV，字段为 question / reference_answer / answer
- 输出：逐条结果 CSV + 汇总报告 JSON + 汇总 Markdown
- 并发打分、失败重试、JSON 解析容错

依赖：
    pip install openai

用法示例：
    # 方式一：环境变量驱动
    export LLM_API_KEY=sk-xxx
    export LLM_API_BASE=https://api.deepseek.com
    export LLM_MODEL_NAME=deepseek-chat
    python llm_judge_eval.py --input samples.jsonl --output results/

    # 方式二：命令行参数优先
    python llm_judge_eval.py --input samples.csv --output results/ \
        --api-key sk-xxx --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --model qwen-plus --workers 4 --max-samples 100

Windows PowerShell 下设置环境变量的等价写法：
    $env:LLM_API_KEY='sk-xxx'
    $env:LLM_API_BASE='https://api.deepseek.com'
    $env:LLM_MODEL_NAME='deepseek-chat'
    python llm_judge_eval.py --input samples.jsonl --output results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print(
        "[错误] 缺少 openai 依赖，请先执行：pip install openai",
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================
# 一、打分 Prompt 设计
# ============================================================

# 系统提示词：定义裁判角色、评分维度、评分标准与输出格式
JUDGE_SYSTEM_PROMPT = """你是一位严谨、公正、经验丰富的大模型评测专家，负责对模型的开放问答答案进行打分。你的评分必须客观、可复现，严格依据评分标准执行，不受模型名称、回答风格或长度影响。

# 评分维度（每项 0-10 分）

1. relevance（相关性）：答案是否直接回应问题，是否答非所问或偏离主题。
2. correctness（正确性）：答案中的事实、推理和结论是否正确，是否存在明显错误。
3. completeness（完整性）：答案是否覆盖问题的全部关键点，是否有重大遗漏。
4. clarity（清晰性）：答案结构是否清晰、表达是否通顺、是否易于理解。

# 评分标准（各维度通用）

- 9-10 分：完全符合该维度要求，无明显瑕疵。
- 7-8 分：基本符合，仅有轻微不足。
- 5-6 分：部分符合，存在明显不足或部分错误。
- 3-4 分：较大程度不符合，存在严重错误或重大遗漏。
- 0-2 分：基本不符合，答非所问或完全错误。

# 总分计算

- total = round(0.35 * relevance + 0.35 * correctness + 0.20 * completeness + 0.10 * clarity, 1)
- 总分范围 0-10，保留 1 位小数。

# 输出要求（必须严格遵守）

- 只输出一个 JSON 对象，不要输出任何其他文字、解释或 Markdown 代码块标记。
- JSON 结构如下：
{"scores": {"relevance": 8, "correctness": 7, "completeness": 6, "clarity": 9}, "total": 7.5, "reason": "简要说明扣分原因与优点，不超过 150 字。"}
"""

# 用户模板：注入单个样本，让裁判评分
JUDGE_USER_TEMPLATE = """请对以下模型回答进行评分。

## 问题
{question}

## 参考答案（评分参考，非唯一标准）
{reference_answer}

## 待评模型回答
{answer}

请严格按照系统提示词中的评分维度、评分标准和输出格式给出评分。"""


# ============================================================
# 二、数据结构
# ============================================================


@dataclass
class Sample:
    """单个评测样本"""

    question: str
    reference_answer: str = ""
    answer: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    """单个样本的裁判打分结果"""

    sample: Sample
    scores: Dict[str, float] = field(default_factory=dict)
    total: Optional[float] = None
    reason: str = ""
    ok: bool = False
    error: str = ""


# ============================================================
# 三、样本加载
# ============================================================


def load_samples(input_path: str) -> List[Sample]:
    """加载 JSONL 或 CSV 格式的评测样本。

    JSONL：每行一个 JSON 对象，支持字段 question / reference_answer / answer
    CSV：表头必须包含 question，建议包含 reference_answer / answer
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    suffix = path.suffix.lower()
    samples: List[Sample] = []

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[警告] 第 {line_no} 行 JSON 解析失败，已跳过：{e}")
                    continue
                samples.append(_item_to_sample(item))
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "question" not in reader.fieldnames:
                raise ValueError("CSV 表头必须包含 question 列")
            for row in reader:
                samples.append(_item_to_sample(row))
    else:
        raise ValueError("仅支持 .jsonl 或 .csv 格式输入")

    if not samples:
        raise ValueError("输入文件中没有有效样本")
    return samples


def _item_to_sample(item: Dict[str, Any]) -> Sample:
    """将一条输入记录转换为 Sample"""
    question = str(item.get("question", "")).strip()
    if not question:
        raise ValueError("样本缺少 question 字段")
    meta = {k: v for k, v in item.items() if k not in ("question", "reference_answer", "answer")}
    return Sample(
        question=question,
        reference_answer=str(item.get("reference_answer", "") or "").strip(),
        answer=str(item.get("answer", "") or "").strip(),
        meta=meta,
    )


# ============================================================
# 四、裁判调用与结果解析
# ============================================================

# 兜底：当模型不返回 JSON 时，用正则提取分数
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_DIM_KEYS = ("relevance", "correctness", "completeness", "clarity")
_DIM_ALIAS = {
    "relevance": ("relevance", "相关性", "相关"),
    "correctness": ("correctness", "正确性", "正确"),
    "completeness": ("completeness", "完整性", "完整"),
    "clarity": ("clarity", "清晰性", "清晰"),
}


def build_messages(sample: Sample) -> List[Dict[str, str]]:
    """组装裁判调用消息"""
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=sample.question,
        reference_answer=sample.reference_answer or "（未提供）",
        answer=sample.answer or "（模型未输出）",
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_score(text: str) -> Dict[str, Any]:
    """解析裁判输出，优先 JSON，失败时正则兜底。

    返回结构：{"scores": {...}, "total": float|None, "reason": str}
    """
    text = (text or "").strip()
    # 去掉可能的 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if m:
        text = m.group(1).strip()

    result: Dict[str, Any] = {"scores": {}, "total": None, "reason": ""}

    # 尝试 JSON 解析
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            scores = data.get("scores") or data
            if isinstance(scores, dict):
                for dim in _DIM_KEYS:
                    val = scores.get(dim)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        result["scores"][dim] = float(val)
            total = data.get("total")
            if isinstance(total, (int, float)) and not isinstance(total, bool):
                result["total"] = float(total)
            reason = data.get("reason", "")
            if isinstance(reason, str):
                result["reason"] = reason.strip()
            return result
    except json.JSONDecodeError:
        pass

    # 正则兜底：寻找 "维度名": 数字 或 "维度名：数字"
    for dim in _DIM_KEYS:
        for alias in _DIM_ALIAS[dim]:
            pat = re.compile(
                rf'["\']?{re.escape(alias)}["\']?\s*[:：]\s*(\d+(?:\.\d+)?)'
            )
            mm = pat.search(text)
            if mm:
                val = float(mm.group(1))
                result["scores"][dim] = min(10.0, val)
                break

    # 总分兜底
    m_total = re.search(r'["\']?total["\']?\s*[:：]\s*(\d+(?:\.\d+)?)', text)
    if m_total:
        result["total"] = min(10.0, float(m_total.group(1)))
    # 理由提取：取末尾一段文字
    reason_m = re.search(r'["\']?reason["\']?\s*[:：]\s*["\']?(.*?)["\']?\s*$', text, re.S)
    if reason_m:
        result["reason"] = reason_m.group(1).strip()

    return result


def compute_total(scores: Dict[str, float]) -> float:
    """按加权公式计算总分（与 Prompt 中一致，保留 1 位小数）"""
    if not scores:
        return 0.0
    w = {"relevance": 0.35, "correctness": 0.35, "completeness": 0.20, "clarity": 0.10}
    total = sum(scores.get(k, 0.0) * w.get(k, 0.0) for k in _DIM_KEYS)
    # 用 Decimal 做四舍五入，避免二进制浮点误差导致 round(7.35,1) 不稳定
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(str(total)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def call_judge(
    client: OpenAI,
    model: str,
    sample: Sample,
    timeout: float = 60.0,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> JudgeResult:
    """调用裁判模型打分，失败自动重试"""
    result = JudgeResult(sample=sample)
    messages = build_messages(sample)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("裁判模型返回空内容")
            parsed = parse_score(text)

            if parsed["scores"]:
                result.scores = parsed["scores"]
            if parsed["total"] is None and result.scores:
                result.total = compute_total(result.scores)
            else:
                result.total = parsed["total"]
            result.reason = parsed["reason"]
            result.ok = bool(result.scores or result.total is not None)
            if result.ok:
                return result
            raise ValueError("无法从裁判输出中解析出分数")
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s 退避
                print(f"[重试] 第 {attempt} 次失败：{e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                result.ok = False
                result.error = str(e)
    return result


# ============================================================
# 五、并发评测与汇总
# ============================================================


def judge_all(
    client: OpenAI,
    model: str,
    samples: List[Sample],
    workers: int = 4,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> List[JudgeResult]:
    """并发评测所有样本，按输入顺序返回结果"""
    results: List[Optional[JudgeResult]] = [None] * len(samples)

    def _judge_one(i: int, s: Sample) -> tuple[int, JudgeResult]:
        return i, call_judge(client, model, s, timeout, max_retries)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_judge_one, i, s) for i, s in enumerate(samples)]
        done = 0
        for fut in as_completed(futures):
            i, res = fut.result()
            results[i] = res
            done += 1
            status = "OK" if res.ok else f"FAIL({res.error[:40]})"
            print(f"[进度] {done}/{len(samples)}  {status}")
    return [r for r in results if r is not None]


def summarize(results: List[JudgeResult]) -> Dict[str, Any]:
    """汇总统计"""
    ok_results = [r for r in results if r.ok]

    def _mean(keys: List[str]) -> Optional[float]:
        vals = [r.total for r in ok_results if r.total is not None] if not keys else []
        if keys:
            vals = []
            for r in ok_results:
                vals.append(sum(r.scores.get(k, 0.0) for k in keys) / len(keys))
        return round(statistics.mean(vals), 2) if vals else None

    summary: Dict[str, Any] = {
        "total_samples": len(results),
        "judged_ok": len(ok_results),
        "judged_fail": len(results) - len(ok_results),
        "score": _mean([]),
        "dims": {},
    }
    for dim in _DIM_KEYS:
        vals = [r.scores.get(dim) for r in ok_results if dim in r.scores]
        summary["dims"][dim] = {
            "mean": round(statistics.mean(vals), 2) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
        }

    totals = [r.total for r in ok_results if r.total is not None]
    if totals:
        summary["score_min"] = min(totals)
        summary["score_max"] = max(totals)
        # 分布：0-2 / 2-4 / 4-6 / 6-8 / 8-10
        buckets = {"0-2": 0, "2-4": 0, "4-6": 0, "6-8": 0, "8-10": 0}
        for t in totals:
            if t < 2:
                buckets["0-2"] += 1
            elif t < 4:
                buckets["2-4"] += 1
            elif t < 6:
                buckets["4-6"] += 1
            elif t < 8:
                buckets["6-8"] += 1
            else:
                buckets["8-10"] += 1
        summary["distribution"] = buckets

    # 最好 / 最差样本
    if ok_results:
        valid = [(r, r.total) for r in ok_results if r.total is not None]
        if valid:
            valid.sort(key=lambda x: x[1])
            worst = valid[0][0]
            best = valid[-1][0]
            summary["best_sample"] = {
                "question": best.sample.question[:100],
                "total": best.total,
            }
            summary["worst_sample"] = {
                "question": worst.sample.question[:100],
                "total": worst.total,
            }
    return summary


# ============================================================
# 六、输出
# ============================================================


def write_results(results: List[JudgeResult], out_dir: Path, summary: Dict[str, Any]) -> Path:
    """写入逐条结果 CSV，返回文件路径"""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "judge_results.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "reference_answer", "answer", "total", "relevance", "correctness", "completeness", "clarity", "reason", "ok", "error"])
        for r in results:
            writer.writerow([
                r.sample.question,
                r.sample.reference_answer,
                r.sample.answer,
                r.total if r.total is not None else "",
                r.scores.get("relevance", ""),
                r.scores.get("correctness", ""),
                r.scores.get("completeness", ""),
                r.scores.get("clarity", ""),
                r.reason,
                "1" if r.ok else "0",
                r.error,
            ])
    return csv_path


def write_summary(summary: Dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """写入汇总 JSON 与 Markdown，返回 (json_path, md_path)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_path = out_dir / "summary.md"
    lines = [
        "# LLM-as-Judge 评测汇总",
        "",
        f"- 评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本总数：{summary.get('total_samples')}",
        f"- 评测成功：{summary.get('judged_ok')}",
        f"- 评测失败：{summary.get('judged_fail')}",
        f"- 平均总分：{summary.get('score')}",
        "",
        "## 各维度得分",
        "",
        "| 维度 | 均值 | 最低 | 最高 |",
        "|---|---|---|---|",
    ]
    for dim, v in summary.get("dims", {}).items():
        lines.append(f"| {dim} | {v.get('mean')} | {v.get('min')} | {v.get('max')} |")
    lines.append("")
    if summary.get("distribution"):
        lines.append("## 总分分布")
        lines.append("")
        lines.append("| 区间 | 数量 |")
        lines.append("|---|---|")
        for bucket, cnt in summary["distribution"].items():
            lines.append(f"| {bucket} | {cnt} |")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return json_path, md_path


# ============================================================
# 七、主入口
# ============================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge 自动化评测脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="评测样本文件路径（.jsonl 或 .csv）")
    parser.add_argument("--output", default="results", help="输出目录")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""), help="API Key（默认取环境变量 LLM_API_KEY）")
    parser.add_argument("--base-url", default=os.environ.get("LLM_API_BASE", ""), help="API Base URL（默认取环境变量 LLM_API_BASE）")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL_NAME", ""), help="裁判模型名（默认取环境变量 LLM_MODEL_NAME）")
    parser.add_argument("--workers", type=int, default=4, help="并发数")
    parser.add_argument("--max-samples", type=int, default=0, help="最多评测样本数（0=全部）")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次请求超时（秒）")
    parser.add_argument("--max-retries", type=int, default=3, help="失败重试次数")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 校验配置
    missing = []
    if not args.api_key:
        missing.append("API Key（--api-key 或环境变量 LLM_API_KEY）")
    if not args.base_url:
        missing.append("Base URL（--base-url 或环境变量 LLM_API_BASE）")
    if not args.model:
        missing.append("模型名（--model 或环境变量 LLM_MODEL_NAME）")
    if missing:
        print("[错误] 缺少必要配置：" + "；".join(missing), file=sys.stderr)
        print("示例：python llm_judge_eval.py --input samples.jsonl --output results/ --api-key sk-xxx --base-url https://api.deepseek.com --model deepseek-chat")
        return 1

    try:
        samples = load_samples(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    print(f"[信息] 加载 {len(samples)} 个样本，开始评测（模型：{args.model}）")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    t0 = time.time()
    results = judge_all(
        client=client,
        model=args.model,
        samples=samples,
        workers=args.workers,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    elapsed = time.time() - t0
    print(f"[信息] 评测完成，耗时 {elapsed:.1f}s")

    summary = summarize(results)
    out_dir = Path(args.output)
    csv_path = write_results(results, out_dir, summary)
    json_path, md_path = write_summary(summary, out_dir)

    print("\n========== 评测汇总 ==========")
    print(f"样本总数：{summary['total_samples']}，成功：{summary['judged_ok']}，失败：{summary['judged_fail']}")
    print(f"平均总分：{summary.get('score')}")
    for dim, v in summary.get("dims", {}).items():
        print(f"  {dim}: mean={v.get('mean')} min={v.get('min')} max={v.get('max')}")
    print("==============================")
    print(f"逐条结果：{csv_path}")
    print(f"汇总 JSON：{json_path}")
    print(f"汇总 Markdown：{md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
