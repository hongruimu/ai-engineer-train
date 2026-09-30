"""使用同一份 V4 query 对比 BM25 与本地向量检索。"""

from __future__ import annotations

import json

import ablation as A
import retrieval as R


METRIC_KEYS = ("recall", "precision", "rr")


def classify(keyword_row: dict, vector_row: dict) -> str:
    keyword_metrics = tuple(keyword_row[key] for key in METRIC_KEYS)
    vector_metrics = tuple(vector_row[key] for key in METRIC_KEYS)

    if vector_metrics == keyword_metrics:
        return "持平"
    if all(vector >= keyword for vector, keyword in zip(vector_metrics, keyword_metrics)):
        return "改善"
    if all(vector <= keyword for vector, keyword in zip(vector_metrics, keyword_metrics)):
        return "退化"
    return "混合"


def print_summary(keyword_report: dict, vector_report: dict) -> None:
    print("=" * 78)
    print("Vector 独立评测 | 同一 V4 query | k=3")
    print("=" * 78)
    print(f"{'Retriever':<24}{'Recall':>10}{'Precision':>12}{'MRR':>10}")
    print("-" * 78)
    for name, report in (("BM25 V4", keyword_report), ("Vector", vector_report)):
        print(
            f"{name:<24}"
            f"{report['mean_recall_at_k']:>10.4f}"
            f"{report['mean_precision_at_k']:>12.4f}"
            f"{report['mrr']:>10.4f}"
        )


def print_per_query(keyword_report: dict, vector_report: dict) -> None:
    keyword_rows = {row["query_id"]: row for row in keyword_report["per_query"]}
    vector_rows = {row["query_id"]: row for row in vector_report["per_query"]}
    counts = {"改善": 0, "退化": 0, "持平": 0, "混合": 0}

    print("\n逐查询对比")
    print("-" * 78)
    for query_id in keyword_rows:
        keyword_row = keyword_rows[query_id]
        vector_row = vector_rows[query_id]
        result = classify(keyword_row, vector_row)
        counts[result] += 1

        print(f"{query_id} · {result} · expected={keyword_row['expected']}")
        print(
            f"  BM25   R={keyword_row['recall']:.3f} "
            f"P={keyword_row['precision']:.3f} RR={keyword_row['rr']:.3f} "
            f"top3={keyword_row['got_top_k']}"
        )
        print(
            f"  Vector R={vector_row['recall']:.3f} "
            f"P={vector_row['precision']:.3f} RR={vector_row['rr']:.3f} "
            f"top3={vector_row['got_top_k']}"
        )

    print("-" * 78)
    print(
        "分类统计："
        + " | ".join(f"{name}={count}" for name, count in counts.items())
    )


def main() -> None:
    docs = R.load_corpus()
    golden = json.loads(R.GOLDEN_FILE.read_text("utf-8"))
    keyword = R.KeywordRetriever(docs)
    vector = R.VectorRetriever(docs)
    print(
        f"Vector 文档缓存：hits={vector.cache_hits}，"
        f"misses={vector.cache_misses}，file={vector.cache_file}"
    )
    evaluate_args = {
        "golden": golden,
        "k": R.TOP_K,
        "n_docs": len(docs),
        "query_builder": A.v4_your_strategy,
    }

    keyword_report = R.evaluate(keyword, **evaluate_args)
    vector_report = R.evaluate(vector, **evaluate_args)

    print_summary(keyword_report, vector_report)
    print_per_query(keyword_report, vector_report)


if __name__ == "__main__":
    main()
