"""Circle 2 主体脚手架 —— 检索层（Retrieval）与 recall@k 评测。

本圈核心问题：
    给定结构化 ProblemModel，如何检索出真正相关的 Solution / Capability / 历史案例？

已搭好的机械部分（你不用动）：
  - Document 数据结构、load_corpus
  - tokenize：中文字符 bigram + ASCII 词（纯标准库，无需分词器）
  - BM25：确定性关键词检索基线（已实现，可直接跑出数）
  - recall_at_k / evaluate：评测器（已实现）
  - self_check：用 corpus/solutions.json + golden.json 跑出基线分数

你要填的关键点（标了 TODO）：
  - build_query：把结构化 ProblemModel 变成检索 query —— 本圈灵魂，即 Context Engineering
  - VectorRetriever / HybridRetriever：Stage 2，需要 embedding API
  - corpus 与 golden set：由你写（格式已给样例）

运行（只用标准库）：
    cd circle-02 && python3 retrieval.py
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
CORPUS_FILE = HERE / "corpus" / "solutions.json"
GOLDEN_FILE = HERE / "golden.json"

TOP_K = 3   # 必须显著小于语料篇数，否则 recall@k 恒为 1.0（见 evaluate 注释）
            # 语料已扩到 23 篇。取 3 也是真实 RAG 里 Context 注入的常见条数（token 预算有限）


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────
@dataclass
class Document:
    doc_id: str
    text: str                                   # 可检索正文
    metadata: dict = field(default_factory=dict)  # domain / capability / deliverable 等，可做过滤


def load_corpus(path: pathlib.Path = CORPUS_FILE) -> list[Document]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text("utf-8"))
    return [Document(d["doc_id"], d["text"], d.get("metadata", {})) for d in raw]


# ─────────────────────────────────────────────────────────────
# 分词（纯标准库）
# ─────────────────────────────────────────────────────────────
# 常用的中日韩统一表意文字
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
# Ascii word
_ASCII_WORD = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文按 unigram + bigram，英文数字按词。

    为什么用 bigram：中文没有空格分词，标准库拿不到分词器。
    bigram 是纯字符级、确定性的近似，足够做关键词基线。
    """
    tokens: list[str] = []
    for w in _ASCII_WORD.findall(text.lower()):
        tokens.append(w)
    for run in _CJK_RUN.findall(text):
        tokens.extend(run)
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


# ─────────────────────────────────────────────────────────────
# BM25 关键词检索基线（已实现）
# ─────────────────────────────────────────────────────────────
class BM25:
    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d.text)) for d in docs]
        self.dl = [sum(c.values()) for c in self.tf]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 0.0
        self.N = len(docs)
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {
            t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()
        }

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
        if not self.docs:
            return []
        qt = tokenize(query)
        results = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            dl = self.dl[i]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 1))
            for t in qt:
                tf = self.tf[i].get(t)
                if not tf:
                    continue
                score += self.idf.get(t, 0.0) * (tf * (self.k1 + 1)) / (tf + denom_norm)
            results.append((doc.doc_id, score))
        results.sort(key=lambda x: (-x[1], x[0]))
        return results[:top_k]


class KeywordRetriever:
    """确定性关键词检索。你的基线，已可用。"""

    name = "keyword(BM25)"

    def __init__(self, docs: list[Document]):
        self.index = BM25(docs)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        return [doc_id for doc_id, _ in self.index.search(query, top_k)]


# ─────────────────────────────────────────────────────────────
# 【核心关键点 1 —— 你来填】
# ─────────────────────────────────────────────────────────────
def build_query(problem_model: dict) -> str:
    """把结构化 ProblemModel 变成一条检索 query。这是本圈的灵魂。

    背景：你 Circle 1 已经能产出结构化 ProblemModel（industry / business_function /
    scenario / pain_points / desired_outcomes / constraints / ...）。
    但检索接口要的是一段文本或一组关键词 —— 中间这一步就是
    **Context Engineering：Context 不是输入，是计算出来的。**

    你要决定的：
      1) 用哪些字段进 query？全部塞进去，还是只用 pain_points + desired_outcomes？
      2) explicit / inferred 要不要区别对待？
         （Circle 1 的教训：confidence 0.2 的 open_question 该不该影响检索？）
      3) constraints 里的 "unknown" 要不要进 query？
         （把 unknown 当关键词检索，会召回一堆无关文档 —— 这是噪声注入）
      4) 字段之间要不要加权？BM25 只吃一段文本，
         但你可以用「重复重要字段」来做隐式加权 —— 这算 hack 还是合理手段？说清理由。

    这个函数决定了召回率的天花板：query 构造得再好的检索器也救不了烂 query。
    """
    parts = []
    for key, value in problem_model.items():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    parts.append(str(item.get("description") or item.get("value") or ""))
                else:
                    parts.append(str(item))
    return " ".join(p for p in parts if p)


# ─────────────────────────────────────────────────────────────
# 【核心关键点 2 —— Stage 2 才做，现在只留接口】
# ─────────────────────────────────────────────────────────────
class VectorRetriever:
    """语义检索。Stage 2。

    需要一个 embedding 服务（阿里云百炼 text-embedding-v3 / OpenAI text-embedding-3-small 等）。
    DeepSeek 目前不提供 embedding 接口，所以 Stage 2 你可能要换一个 key。

    实现要点（想清楚再写）：
      - 文档侧 embedding 可以离线算好缓存（Circle 1 你已见过 prompt_cache 命中）
      - 相似度用 cosine；top_k 截断
      - 与前置题结论呼应：embedding 度量的是**话题相似**，不是**逻辑蕴含**。
        对 F13 那类 entailment 断言它会给假阳性。检索场景里话题相似通常够用，
        但你要能说清这个区别。
    """

    name = "vector(cosine)"

    def __init__(self, docs: list[Document]):
        raise NotImplementedError("Stage 2：语义检索由你实现")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        raise NotImplementedError


class HybridRetriever:
    """关键词 + 语义融合（RRF 或加权）。Stage 2。

    为什么需要融合：关键词擅长精确术语（"ERP""OCR"），语义擅长同义改写
    （"比价太慢" vs "报价自动化分析"）。两者失败模式互补。
    """

    name = "hybrid(keyword+vector)"

    def __init__(self, docs: list[Document]):
        raise NotImplementedError("Stage 2：由你实现")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────
# 评测器（已实现）
# ─────────────────────────────────────────────────────────────
def recall_at_k(retrieved: list[str], relevant: list[str], k: int = TOP_K) -> float:
    """召回率：该找到的文档，有多少落在 top_k 里。"""
    top = set(retrieved[:k])
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(top & rel) / len(rel)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int = TOP_K) -> float:
    """精确率：top_k 里有多少是真该找到的。

    为什么必须加这个指标：recall@k 只惩罚「漏」，不惩罚「滥」。
    语料 6 篇、k=5 时，随便返回 5 篇，唯一那篇正例几乎必然被覆盖 → recall 恒为 1.0。
    这个满分不携带任何信息。precision 才暴露「排序有没有把噪声压下去」。
    """
    top = retrieved[:k]
    rel = set(relevant)
    if not top:
        return 0.0
    return sum(1 for d in top if d in rel) / len(top)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """MRR 的分项：第一篇正例排在第几位，取倒数。衡量「排序质量」。"""
    rel = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def evaluate(retriever, golden: list[dict], k: int = TOP_K, n_docs: int | None = None,
             query_builder=None) -> dict:
    """query_builder: 可注入的 ProblemModel → query 函数，默认用模块级 build_query。

    之所以要可注入：消融实验需要对同一份语料/golden 跑多个 query 构造策略做对照。
    """
    build = query_builder or build_query
    # ── 防呆：k 逼近语料规模时，recall@k 是恒真指标，必须先拦住 ──
    if n_docs is not None and n_docs <= k * 2:
        raise ValueError(
            f"评测无判别力：语料只有 {n_docs} 篇，而 k={k}。"
            f"此时任意检索器都会得到接近满分的 recall —— 指标测的是语料规模，不是检索能力。\n"
            f"修法二选一：① 把语料扩到至少 {k * 3} 篇；② 把 TOP_K 降到 {max(1, n_docs // 3)}。"
        )

    recs, precs, rrs, per_query = [], [], [], []
    for case in golden:
        query = build(case["problem_model"]) if "problem_model" in case else case.get("query_text", "")
        got = retriever.retrieve(query, k)
        rel = case["relevant_doc_ids"]
        r, p, rr = recall_at_k(got, rel, k), precision_at_k(got, rel, k), reciprocal_rank(got, rel)
        recs.append(r)
        precs.append(p)
        rrs.append(rr)
        per_query.append(
            {
                "query_id": case["query_id"],
                "recall": round(r, 3),
                "precision": round(p, 3),
                "rr": round(rr, 3),
                "expected": rel,
                "got_top_k": got,
            }
        )
    n = len(recs) or 1
    return {
        "retriever": getattr(retriever, "name", type(retriever).__name__),
        "mean_recall_at_k": round(sum(recs) / n, 4),
        "mean_precision_at_k": round(sum(precs) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        "k": k,
        "n_queries": len(recs),
        "perfect": sum(1 for s in recs if s == 1.0),
        "zero": sum(1 for s in recs if s == 0.0),
        "per_query": per_query,
    }


def self_check() -> None:
    docs = load_corpus()
    if not docs:
        print(f"⚠️  语料为空：{CORPUS_FILE} 不存在或没有文档。先写 corpus。")
        return
    golden = json.loads(GOLDEN_FILE.read_text("utf-8")) if GOLDEN_FILE.exists() else []
    if not golden:
        print(f"⚠️  golden set 为空：{GOLDEN_FILE} 不存在。先写 golden.json。")
        print(f"   （已加载语料 {len(docs)} 篇）")
        return

    print("=" * 70)
    print(f"语料 {len(docs)} 篇 | golden {len(golden)} 条查询 | k={TOP_K}")
    print("=" * 70)

    report = evaluate(KeywordRetriever(docs), golden, n_docs=len(docs))
    print(f"\n【{report['retriever']}】")
    print(f"  mean recall@{report['k']}    = {report['mean_recall_at_k']}")
    print(f"  mean precision@{report['k']} = {report['mean_precision_at_k']}   ← 主指标，recall 会被语料规模骗")
    print(f"  MRR                    = {report['mrr']}")
    print(f"  满分查询 {report['perfect']}/{report['n_queries']} | 零分查询 {report['zero']}/{report['n_queries']}")
    for row in report["per_query"]:
        flag = "✅" if row["recall"] == 1.0 else ("❌" if row["recall"] == 0.0 else "⚠️")
        print(
            f"  {flag} {row['query_id']}: recall={row['recall']} P={row['precision']} "
            f"期望={row['expected']} 实际={row['got_top_k']}"
        )

    print("\n" + "─" * 70)
    print("读这个基线分数的方式：")
    print("  · recall 高但 precision 低 → 正例被召回了，但淹没在噪声里。")
    print("    这在真实 RAG 里等于失败：塞进 Context 的全是干扰，还会挤占 token 预算。")
    print("  · recall 也低 → 关键词真的抓不住，这才是引入向量检索的理由。")
    print("  · 逐条看 per_query 里排在正例前面的那些 doc：它们是靠哪个词被召回的？")
    print("    是 query 里混进了噪声字段（unknown / 低置信度推断），还是语料本身词面重叠？")
    print("─" * 70)


if __name__ == "__main__":
    self_check()
