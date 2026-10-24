# -*- coding: utf-8 -*-
"""dry-run 测试：模拟裁判输出，验证 load_samples / parse_score / summarize / 输出逻辑"""
import sys, json
from pathlib import Path

sys.path.insert(0, r"C:\Users\X1 Yoga\Doubao\chats\2026-10-23\new-chat")
import llm_judge_eval as j

# 1. 加载样本
samples = j.load_samples(r"C:\Users\X1 Yoga\Doubao\chats\2026-10-23\new-chat\samples_demo.jsonl")
assert len(samples) == 2, f"样本数应为 2，实际 {len(samples)}"
assert samples[0].question.startswith("什么是"), samples[0].question
print("[OK] load_samples: 2 个样本加载成功")

# 2. 消息组装
msgs = j.build_messages(samples[0])
assert msgs[0]["role"] == "system" and "评分维度" in msgs[0]["content"]
assert samples[0].question in msgs[1]["content"]
print("[OK] build_messages: 系统提示词与用户模板组装正确")

# 3. 解析正常 JSON 输出
text_ok = '{"scores": {"relevance": 8, "correctness": 7, "completeness": 6, "clarity": 9}, "total": 7.5, "reason": "回答正确且相关，完整性略不足。"}'
p = j.parse_score(text_ok)
assert p["scores"]["relevance"] == 8.0 and p["total"] == 7.5
assert "完整性" in p["reason"]
print("[OK] parse_score: 标准 JSON 解析正确")

# 4. 解析带 ```json 包裹与中文冒号
text_wrapped = '```json\n{"scores": {"relevance": 9, "correctness": 8, "completeness": 7, "clarity": 8}, "total": 8.1, "reason": "整体优秀"}\n```'
p2 = j.parse_score(text_wrapped)
assert p2["scores"]["correctness"] == 8.0 and p2["total"] == 8.1
print("[OK] parse_score: 代码块包裹解析正确")

# 5. 正则兜底：模型返回非 JSON 文本
text_fallback = "相关性：7分；正确性：6分；完整性：5分；清晰性：8分。理由：部分要点遗漏。"
p3 = j.parse_score(text_fallback)
assert p3["scores"].get("relevance") == 7.0, p3["scores"]
assert p3["scores"].get("clarity") == 8.0
assert p3["total"] is None  # 无 total 时由 compute_total 计算
print("[OK] parse_score: 非 JSON 正则兜底正确")

# 6. 加权总分计算（与 Prompt 公式一致，保留 1 位小数）
total = j.compute_total({"relevance": 8, "correctness": 7, "completeness": 6, "clarity": 9})
expected = round(8*0.35 + 7*0.35 + 6*0.20 + 9*0.10, 1)
assert abs(total - expected) < 1e-6, f"{total} != {expected}"
print(f"[OK] compute_total: 加权公式正确（示例总分 {total}）")

# 7. 汇总统计（用解析结果构造 JudgeResult）
results = []
for s, txt in zip(samples, [text_ok, text_wrapped]):
    parsed = j.parse_score(txt)
    r = j.JudgeResult(sample=s)
    r.scores = parsed["scores"]
    r.total = parsed["total"] or j.compute_total(parsed["scores"])
    r.reason = parsed["reason"]
    r.ok = True
    results.append(r)

summary = j.summarize(results)
assert summary["total_samples"] == 2 and summary["judged_ok"] == 2
assert summary["score"] is not None
assert "distribution" in summary and "best_sample" in summary
print(f"[OK] summarize: 汇总正确（平均总分 {summary['score']}，分布 {summary['distribution']}）")

# 8. 输出文件写入
out_dir = Path(r"C:\Users\X1 Yoga\Doubao\chats\2026-10-23\new-chat\dryrun_out")
csv_path = j.write_results(results, out_dir, summary)
jp, mp = j.write_summary(summary, out_dir)
assert csv_path.exists() and jp.exists() and mp.exists()
csv_text = csv_path.read_text(encoding="utf-8-sig")
assert "question" in csv_text and "total" in csv_text
print(f"[OK] 输出文件写入成功：{csv_path.name} / {jp.name} / {mp.name}")

# 9. 错误样本处理
r_fail = j.JudgeResult(sample=samples[0]); r_fail.ok = False; r_fail.error = "timeout"
results2 = results + [r_fail]
s2 = j.summarize(results2)
assert s2["judged_fail"] == 1 and s2["total_samples"] == 3
print("[OK] 失败样本统计正确")

print("\n========== 全部 dry-run 测试通过 ==========")
