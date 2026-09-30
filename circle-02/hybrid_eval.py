"""使用同一份 V4 query 对比 BM25、Vector 与 RRF Hybrid。"""

from __future__ import annotations

import json

import ablation as A
import retrieval as R


METRIC_KEYS = ("recall", "precision", "rr")


def compare(base_row: dict, candidate_row: dict) -> str:
    """比较同一查询的三个指标，返回候选方案相对基线的变化。"""
    base_metrics = tuple(base_row[key] for key in METRIC_KEYS)
    candidate_metrics = tuple(candidate_row[key] for key in METRIC_KEYS)

    if candidate_metrics == base_metrics:
        return "持平"
    if all(candidate >= base for candidate, base in zip(candidate_metrics, base_metrics)):
        return "改善"
    if all(candidate <= base for candidate, base in zip(candidate_metrics, base_metrics)):
        return "退化"
    return "混合"


def print_summary(reports: list[tuple[str, dict]]) -> None:
    print("=" * 88)
    print("Hybrid 完整评测 | 同一 V4 query | k=3")
    print("=" * 88)
    print(f"{'Retriever':<24}{'Recall':>10}{'Precision':>12}{'MRR':>10}")
    print("-" * 88)
    for name, report in reports:
        print(
            f"{name:<24}"
            f"{report['mean_recall_at_k']:>10.4f}"
            f"{report['mean_precision_at_k']:>12.4f}"
            f"{report['mrr']:>10.4f}"
        )


def print_per_query(bm25_report: dict, vector_report: dict, hybrid_report: dict) -> None:
    bm25_rows = {row["query_id"]: row for row in bm25_report["per_query"]}
    vector_rows = {row["query_id"]: row for row in vector_report["per_query"]}
    hybrid_rows = {row["query_id"]: row for row in hybrid_report["per_query"]}
    versus_bm25 = {"改善": 0, "退化": 0, "持平": 0, "混合": 0}
    versus_vector = {"改善": 0, "退化": 0, "持平": 0, "混合": 0}

    print("\n逐查询对比")
    print("-" * 88)
    for query_id, bm25_row in bm25_rows.items():
        vector_row = vector_rows[query_id]
        hybrid_row = hybrid_rows[query_id]
        compared_with_bm25 = compare(bm25_row, hybrid_row)
        compared_with_vector = compare(vector_row, hybrid_row)
        versus_bm25[compared_with_bm25] += 1
        versus_vector[compared_with_vector] += 1

        print(
            f"{query_id} · Hybrid vs BM25={compared_with_bm25} "
            f"· Hybrid vs Vector={compared_with_vector} "
            f"· expected={hybrid_row['expected']}"
        )
        for name, row in (
            ("BM25", bm25_row),
            ("Vector", vector_row),
            ("Hybrid", hybrid_row),
        ):
            print(
                f"  {name:<6} R={row['recall']:.3f} "
                f"P={row['precision']:.3f} RR={row['rr']:.3f} "
                f"top3={row['got_top_k']}"
            )

    print("-" * 88)
    print(
        "Hybrid vs BM25："
        + " | ".join(f"{name}={count}" for name, count in versus_bm25.items())
    )
    print(
        "Hybrid vs Vector："
        + " | ".join(f"{name}={count}" for name, count in versus_vector.items())
    )


def main() -> None:
    docs = R.load_corpus()
    golden = json.loads(R.GOLDEN_FILE.read_text("utf-8"))
    bm25 = R.KeywordRetriever(docs)
    vector = R.VectorRetriever(docs)
    hybrid = R.HybridRetriever(docs)
    evaluate_args = {
        "golden": golden,
        "k": R.TOP_K,
        "n_docs": len(docs),
        "query_builder": A.v4_your_strategy,
    }

    bm25_report = R.evaluate(bm25, **evaluate_args)
    vector_report = R.evaluate(vector, **evaluate_args)
    hybrid_report = R.evaluate(hybrid, **evaluate_args)

    print(
        f"Vector 缓存：hits={vector.cache_hits}，misses={vector.cache_misses}；"
        f"Hybrid 内部 Vector 缓存：hits={hybrid.vector_retriever.cache_hits}，"
        f"misses={hybrid.vector_retriever.cache_misses}"
    )
    print_summary(
        [
            ("BM25 V4", bm25_report),
            ("Vector", vector_report),
            ("Hybrid RRF", hybrid_report),
        ]
    )
    print_per_query(bm25_report, vector_report, hybrid_report)


if __name__ == "__main__":
    main()
