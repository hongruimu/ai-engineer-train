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
    """确定性关键词检索。你的基线，已可用。

    Stage 2 之前，它没有「弃权」能力：无论 query 多不相关，BM25 永远返回 k 篇。
    这意味着零正例查询永远走「硬凑」分支（recall=0.0）。
    下面的 score_floor 是给弃权留的接口，但⚠️它有一个陷阱，见 should_abstain。
    """

    name = "keyword(BM25)"

    def __init__(self, docs: list[Document], abstain: bool = False):
        self.index = BM25(docs)
        # abstain: 是否启用弃权。False = 保持原行为（永远返回 k 篇）。
        # True = 调用 should_abstain 逐条过滤，全被丢弃则返回 []。
        self.abstain = abstain

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[str]:
        scored = self.index.search(query, top_k)          # [(doc_id, score), ...]
        if not self.abstain:
            return [doc_id for doc_id, _ in scored]
        # 弃权判定：把 should_abstain 判 True 的结果丢掉，全丢光就返回 []
        kept = [(d, s) for d, s in scored if not self.should_abstain(d, s, scored)]
        return [doc_id for doc_id, _ in kept]

    def should_abstain(self, doc_id: str, score: float, all_scored: list[tuple[str, float]]) -> bool:
        """【Step 8 训练点 (b) —— 你来填策略，我不代写】

        判断这条结果是否该被丢弃（弃权）。返回 True = 丢掉它。

        ⚠️ 已实测结论：**在 BM25 分数上，这个函数无解。不要试图填它。**
        2026-09-29 穷举验证了四种规则，判据是「该保留的查询」与「该丢弃的查询」
        的取值区间是否分离：

            规则                 Q1(该留,5正例)  Q2(该弃)  Q3(该弃)   可行?
            ─────────────────────────────────────────────────────────────
            A ratio  = s1/s2          1.13        1.78      1.04     ❌ 重叠
            A gap    = s1-s2         50.51        4.80      0.30     ⚠️ 见下
            A norm_gap = gap/ntok     0.05        0.18      0.01     ❌ 重叠
            B norm1  = s1/ntok        0.39        0.41      0.34     ❌ 重叠
            C rel_mean = s1/mean      1.99        3.69      2.54     ❌ 重叠
            D 绝对 s1               429.9        11.0       7.9     ⚠️ 见下

        A(ratio) 最反直觉：Q2 的 top1/top2=1.78 比 Q1 的 1.13 还大，
        即「断层更明显」的反而是该弃权的那条。用 gap 判会正好搞反。

        gap 和绝对 s1 看似可行，但它们分离的是 **query 长度**，不是相关性：
        Q1 的 query 是 774 字符/1099 token，Q3 是 12 字符/23 token，长 64 倍。
        BM25 分数随 query 词数累加，所以长 query 必然高分。
        把 Q2/Q3 重建成同规格 ProblemModel 走同一条 build_query 后，
        top1 变成 25.9 / 22.3，与 Q1 的 429.9 仍差 17 倍——差距来自字段填充量。
        → 用 gap/绝对阈值等于「query 短就弃权」，与相关性无关。

        B(归一化) 是最有力的反证：按 token 数归一化后，
        Q2(零正例) 的 0.41 反而 **高于** Q1(5个正例) 的 0.39。
        原因：Q1 的长 query 里大量 token 匹配不到任何文档，摊薄了每 token 得分。
        这说明 BM25 分数里混入了「query 有多少词命中过东西」，
        而这个量与「命中的是不是对的」无关。

        **根本原因**：BM25 度量的是词面重叠，不是语义相关。
        Q2 该弃权，但它召回的 F2 是「同词面不同领域」的干扰文档——
        词面上 F2 确实高度匹配「设备预测性维护」。BM25 报告高分是**正确**的，
        错的是把词面相似当语义相关。任何对词面分数的单调变换都无法恢复语义相关性。

        **所以弃权能力属于 Stage 2**（向量检索 / cross-encoder 重排 / LLM 判定），
        不属于关键词基线。这个函数保留是为 Stage 2 的检索器准备的接口。
        关键词检索器的正确取值就是 abstain=False，且 Q2/Q3 必须保持 ❌——
        那是对真实产品风险的**正确测量**，不是待修复的缺陷。
        """
        return False   # 关键词层无弃权能力，见上方实测结论


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
    # top空代表没有召回内容，rel为空代表真实没有内容，这个召回率值给1代表系统确实召回率高，没有匹配的内容确实没召回
    if not top and not rel:
        return 1.0
    
    # top不为空，但是rel为空，证明召回存在噪声，导致召回率并不高
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
    # 都为空，没有召回任何内容，证明精度就是很高
    if not top and not rel:
        return 1.0
    # rel不为空，但是top为空，证明确实啥也没召回，就是精度很低，给0是合理的
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
    # ── 防呆：未标注（None）不能等同于「零正例」（[]）──
    # [] 是你判断过「确实没有对应方案」；None 是你还没标。
    # 两者混在一起会静默拉低平均分，且看不出来是哪条没标。
    unlabeled = [c.get("query_id", "?") for c in golden if c.get("relevant_doc_ids") is None]
    if unlabeled:
        raise ValueError(
            f"以下查询尚未标注 relevant_doc_ids：{unlabeled}\n"
            f"请用 null（未标注）与 []（已确认零正例）区分两种状态。\n"
            f"运行 python3 label.py 做盲标。"
        )
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
