"""对冻结的 holdout 查询只运行一次 BM25、Vector 与 Hybrid 评测。"""

from __future__ import annotations

import json
import pathlib

import ablation as A
import retrieval as R


HOLDOUT_FILE = R.HERE / "golden_holdout.json"
REPORT_FILE = R.HERE / "holdout_report.json"
EXPECTED_DOCUMENTS = 23
EXPECTED_QUERIES = 15
TOP_K = 3
CANDIDATE_K = 10
RRF_CONSTANT = 60.0
EMBED_MODEL = "qwen3-embedding:0.6b"
METRIC_KEYS = ("recall", "precision", "rr")


def validate_inputs(docs: list[R.Document], golden: list[dict]) -> None:
    if len(docs) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"冻结语料应为 {EXPECTED_DOCUMENTS} 篇，实际为 {len(docs)} 篇"
        )
    if len(golden) != EXPECTED_QUERIES:
        raise ValueError(
            f"冻结测试集应为 {EXPECTED_QUERIES} 条，实际为 {len(golden)} 条"
        )
    unlabeled = [
        case.get("query_id", "?")
        for case in golden
        if case.get("relevant_doc_ids") is None
    ]
    if unlabeled:
        raise ValueError(f"holdout 仍有未标注查询：{unlabeled}")
    if R.TOP_K != TOP_K:
        raise ValueError(f"retrieval.TOP_K 应冻结为 {TOP_K}，实际为 {R.TOP_K}")


def compare(base_row: dict, candidate_row: dict) -> str:
    base_metrics = tuple(base_row[key] for key in METRIC_KEYS)
    candidate_metrics = tuple(candidate_row[key] for key in METRIC_KEYS)
    if candidate_metrics == base_metrics:
        return "持平"
    if all(candidate >= base for candidate, base in zip(candidate_metrics, base_metrics)):
        return "改善"
    if all(candidate <= base for candidate, base in zip(candidate_metrics, base_metrics)):
        return "退化"
    return "混合"


def rows_by_query(report: dict) -> dict[str, dict]:
    return {row["query_id"]: row for row in report["per_query"]}


def has_relevant(row: dict) -> bool:
    return bool(set(row["expected"]) & set(row["got_top_k"]))


def build_diagnostics(
    bm25_report: dict,
    vector_report: dict,
    hybrid_report: dict,
) -> dict:
    bm25_rows = rows_by_query(bm25_report)
    vector_rows = rows_by_query(vector_report)
    hybrid_rows = rows_by_query(hybrid_report)
    vector_only_rescues = []
    bm25_only_protections = []
    hybrid_vs_bm25 = {"改善": [], "退化": [], "持平": [], "混合": []}
    hybrid_vs_vector = {"改善": [], "退化": [], "持平": [], "混合": []}
    zero_positive = {}

    for query_id, bm25_row in bm25_rows.items():
        vector_row = vector_rows[query_id]
        hybrid_row = hybrid_rows[query_id]

        if vector_row["expected"]:
            if has_relevant(vector_row) and not has_relevant(bm25_row):
                vector_only_rescues.append(query_id)
            if has_relevant(bm25_row) and not has_relevant(vector_row):
                bm25_only_protections.append(query_id)

        hybrid_vs_bm25[compare(bm25_row, hybrid_row)].append(query_id)
        hybrid_vs_vector[compare(vector_row, hybrid_row)].append(query_id)

        if not hybrid_row["expected"]:
            zero_positive[query_id] = {
                "bm25_abstained": not bm25_row["got_top_k"],
                "vector_abstained": not vector_row["got_top_k"],
                "hybrid_abstained": not hybrid_row["got_top_k"],
                "bm25_top3": bm25_row["got_top_k"],
                "vector_top3": vector_row["got_top_k"],
                "hybrid_top3": hybrid_row["got_top_k"],
            }

    return {
        "vector_only_rescues": vector_only_rescues,
        "bm25_only_protections": bm25_only_protections,
        "hybrid_vs_bm25": hybrid_vs_bm25,
        "hybrid_vs_vector": hybrid_vs_vector,
        "zero_positive_queries": zero_positive,
    }


def print_summary(reports: list[tuple[str, dict]]) -> None:
    print("=" * 92)
    print("Holdout 一次性冻结评测 | V4 query | k=3 | 23 docs | 15 queries")
    print("=" * 92)
    print(f"{'Retriever':<24}{'Recall':>10}{'Precision':>12}{'MRR':>10}")
    print("-" * 92)
    for name, report in reports:
        print(
            f"{name:<24}"
            f"{report['mean_recall_at_k']:>10.4f}"
            f"{report['mean_precision_at_k']:>12.4f}"
            f"{report['mrr']:>10.4f}"
        )


def print_per_query(reports: list[tuple[str, dict]]) -> None:
    report_rows = {
        name: rows_by_query(report)
        for name, report in reports
    }
    first_report = reports[0][1]

    print("\n逐查询 Top-3")
    print("-" * 92)
    for base_row in first_report["per_query"]:
        query_id = base_row["query_id"]
        print(f"{query_id} · expected={base_row['expected']}")
        for name, _ in reports:
            row = report_rows[name][query_id]
            print(
                f"  {name:<10} R={row['recall']:.3f} "
                f"P={row['precision']:.3f} RR={row['rr']:.3f} "
                f"top3={row['got_top_k']}"
            )


def print_diagnostics(diagnostics: dict) -> None:
    print("\n互补性与弃权诊断")
    print("-" * 92)
    print(f"Vector-only rescues：{diagnostics['vector_only_rescues'] or '无'}")
    print(f"BM25-only protections：{diagnostics['bm25_only_protections'] or '无'}")
    print(f"Hybrid vs BM25：{diagnostics['hybrid_vs_bm25']}")
    print(f"Hybrid vs Vector：{diagnostics['hybrid_vs_vector']}")
    for query_id, row in diagnostics["zero_positive_queries"].items():
        print(
            f"{query_id} 零正例弃权：BM25={row['bm25_abstained']} "
            f"Vector={row['vector_abstained']} Hybrid={row['hybrid_abstained']}"
        )


def main() -> None:
    if REPORT_FILE.exists():
        raise RuntimeError(
            f"冻结 holdout 已评测并生成 {REPORT_FILE.name}，禁止重复运行或据此调参"
        )

    docs = R.load_corpus()
    golden = json.loads(HOLDOUT_FILE.read_text("utf-8"))
    validate_inputs(docs, golden)

    embedding_client = R.OllamaEmbeddingClient(model=EMBED_MODEL)
    bm25 = R.KeywordRetriever(docs)
    vector = R.VectorRetriever(docs, client=embedding_client)
    hybrid = R.HybridRetriever(
        docs,
        vector_client=embedding_client,
        candidate_k=CANDIDATE_K,
        rank_constant=RRF_CONSTANT,
    )
    evaluate_args = {
        "golden": golden,
        "k": TOP_K,
        "n_docs": len(docs),
        "query_builder": A.v4_your_strategy,
    }

    bm25_report = R.evaluate(bm25, **evaluate_args)
    vector_report = R.evaluate(vector, **evaluate_args)
    hybrid_report = R.evaluate(hybrid, **evaluate_args)
    reports = [
        ("BM25 V4", bm25_report),
        ("Vector", vector_report),
        ("Hybrid RRF", hybrid_report),
    ]
    diagnostics = build_diagnostics(
        bm25_report,
        vector_report,
        hybrid_report,
    )
    payload = {
        "evaluation_status": "frozen_holdout_run",
        "config": {
            "golden": HOLDOUT_FILE.name,
            "corpus": str(R.CORPUS_FILE.relative_to(R.HERE)),
            "n_documents": len(docs),
            "n_queries": len(golden),
            "query_builder": "v4_your_strategy",
            "top_k": TOP_K,
            "embedding_model": EMBED_MODEL,
            "candidate_k": CANDIDATE_K,
            "rrf_constant": RRF_CONSTANT,
        },
        "cache": {
            "vector_hits": vector.cache_hits,
            "vector_misses": vector.cache_misses,
            "hybrid_vector_hits": hybrid.vector_retriever.cache_hits,
            "hybrid_vector_misses": hybrid.vector_retriever.cache_misses,
        },
        "reports": {
            "bm25_v4": bm25_report,
            "vector": vector_report,
            "hybrid_rrf": hybrid_report,
        },
        "diagnostics": diagnostics,
    }
    REPORT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        "utf-8",
    )

    print(
        f"Vector 缓存：hits={vector.cache_hits}，misses={vector.cache_misses}；"
        f"Hybrid 内部 Vector 缓存：hits={hybrid.vector_retriever.cache_hits}，"
        f"misses={hybrid.vector_retriever.cache_misses}"
    )
    print_summary(reports)
    print_per_query(reports)
    print_diagnostics(diagnostics)
    print(f"\n完整报告已写入：{pathlib.Path(REPORT_FILE).name}")


if __name__ == "__main__":
    main()
