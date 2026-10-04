from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    zeros = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "per_question": [
            EvalResult(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=gt,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ],
    }
    if not questions:
        return zeros

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        per_question = []
        for _, row in df.iterrows():
            f_val = float(row.get("faithfulness", 0.0) or 0.0)
            ar_val = float(row.get("answer_relevancy", 0.0) or 0.0)
            cp_val = float(row.get("context_precision", 0.0) or 0.0)
            cr_val = float(row.get("context_recall", 0.0) or 0.0)

            f_val = 0.0 if math.isnan(f_val) else f_val
            ar_val = 0.0 if math.isnan(ar_val) else ar_val
            cp_val = 0.0 if math.isnan(cp_val) else cp_val
            cr_val = 0.0 if math.isnan(cr_val) else cr_val

            ctx = row["contexts"]
            if not isinstance(ctx, list):
                ctx = list(ctx)

            per_question.append(
                EvalResult(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    contexts=ctx,
                    ground_truth=str(row["ground_truth"]),
                    faithfulness=round(f_val, 4),
                    answer_relevancy=round(ar_val, 4),
                    context_precision=round(cp_val, 4),
                    context_recall=round(cr_val, 4),
                )
            )

        def _mean_metric(name: str) -> float:
            if name in df:
                vals = [float(v) for v in df[name] if not math.isnan(float(v or 0.0))]
                return round(sum(vals) / len(vals), 4) if vals else 0.0
            val = float(result.get(name, 0.0) or 0.0)
            return 0.0 if math.isnan(val) else round(val, 4)

        return {
            "faithfulness": _mean_metric("faithfulness"),
            "answer_relevancy": _mean_metric("answer_relevancy"),
            "context_precision": _mean_metric("context_precision"),
            "context_recall": _mean_metric("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return zeros


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM tự bịa câu trả lời ngoài tài liệu (LLM hallucinating)",
            "Thắt chặt system prompt, giảm nhiệt độ (temperature) về 0 (Tighten prompt, lower temperature)"
        ),
        "context_recall": (
            "Hệ thống tìm kiếm bỏ sót đoạn văn đúng (Missing relevant chunks)",
            "Cải thiện lại bước cắt đoạn hoặc bổ sung từ khóa BM25 (Improve chunking or add BM25)"
        ),
        "context_precision": (
            "Đoạn văn không liên quan bị xếp lên đầu (Too many irrelevant chunks)",
            "Bổ sung tầng Cross-Encoder reranking hoặc lọc theo metadata (Add reranking or metadata filter)"
        ),
        "answer_relevancy": (
            "Câu trả lời bị lệch trọng tâm câu hỏi (Answer doesn't match question)",
            "Viết lại prompt hướng dẫn mô hình trả lời trực tiếp hơn (Improve prompt template)"
        ),
    }

    scored_items = []
    for res in eval_results:
        metric_scores = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall,
        }
        avg_score = sum(metric_scores.values()) / 4.0
        worst_metric = min(metric_scores, key=lambda k: metric_scores[k])
        worst_score = metric_scores[worst_metric]
        diagnosis, fix = diagnostic_tree.get(worst_metric, ("Unknown error", "Review pipeline"))

        scored_items.append({
            "question": res.question,
            "answer": res.answer,
            "ground_truth": res.ground_truth,
            "avg_score": round(avg_score, 4),
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored_items.sort(key=lambda x: (x["avg_score"], x["score"]))
    return scored_items[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
