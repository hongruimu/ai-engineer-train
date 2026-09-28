"""Circle 2 消融实验 —— query 构造策略对照。

脚手架已经帮你跑出结论性证据，你要做的是：读懂它，然后自己写 V4。

运行：
    cd circle-02 && python3 ablation.py
"""

from __future__ import annotations

import json

import retrieval as R


# ─────────────────────────────────────────────────────────────
# 变体定义：每个都是 ProblemModel → query 的一种策略
# ─────────────────────────────────────────────────────────────
def v0_all_fields(pm: dict) -> str:
    """【你现在的 build_query】所有字段无差别拼接。"""
    return R.build_query(pm)


def v1_drop_unknown(pm: dict) -> str:
    """V0 + 剔除 "unknown" 噪声。

    证据：你的 query 里 "unknown" 出现了 6 次（constraints 的五个 unknown + industry）。
    BM25 会把它当成一个普通检索词，去匹配所有含 unknown 的文档。
    这不是「检索到相关内容」，是「噪声自己匹配自己」。
    """
    text = v0_all_fields(pm)
    return " ".join(w for w in text.split() if w.strip().lower() != "unknown")


def v2_high_confidence_only(pm: dict) -> str:
    """只保留 explicit + confidence>=0.6 的内容。

    对应 Circle 1 的教训：confidence 0.3 的 open_question（"是否希望自动生成比价结论"）
    是你的猜测，不是客户诉求。把它塞进 query，等于让猜测去决定检索方向。
    """
    THRESHOLD = 0.6
    parts = []
    for key in ("industry", "business_function", "scenario"):
        v = pm.get(key)
        if isinstance(v, str) and v.strip().lower() != "unknown":
            parts.append(v)
    for key in ("pain_points", "desired_outcomes", "open_questions", "constraints"):
        for item in pm.get(key, []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("source") == "explicit":
                parts.append(str(item.get("description") or item.get("value") or ""))
            elif (item.get("confidence") or 0) >= THRESHOLD:
                parts.append(str(item.get("description") or item.get("value") or ""))
    for a in pm.get("data_assets", []) or []:
        if not str(a).startswith("【待澄清】"):
            parts.append(str(a))
    for s in pm.get("current_workflow", []) or []:
        if not str(s).startswith("【待澄清】"):
            parts.append(str(s))
    return " ".join(p for p in parts if p.strip())


def v3_scenario_and_function(pm: dict) -> str:
    """极简：只用 scenario + business_function 两个高抽象字段。

    这是一个「反向极端」变体，用来标定另一端。
    如果它和 V0 分数差不多，说明 V0 里那一堆 pain_points 根本没起作用。
    """
    return " ".join(
        str(pm.get(k, "")) for k in ("business_function", "scenario") if str(pm.get(k, "")) != "unknown"
    )


# ─────────────────────────────────────────────────────────────
# 【你来填】V4：你自己的策略
# ─────────────────────────────────────────────────────────────
def v4_your_strategy(pm: dict) -> str:
    """TODO —— 本圈真正的训练点，不许跳过。

    先读完下面的实验输出，再回答这三个问题，然后写实现：

    1) 归因证据显示，C1 排第一靠的是「段 / 的 / 一 / 与 / 价」这类高频单字，
       而不是「报价单解析」「横向比价」这类真正有区分度的词。
       → 你的 tokenizer 同时产 unigram 和 bigram，unigram 让所有中文文档互相"沾边"。
         在 query 侧你能做什么来放大高区分度词的权重？
         
        - 通过

    2) pain_points 用的是客户口语（"耗时间""遗漏"），C1 用的是方案术语
       （"横向比价""归一化""异常报价标记"）。两者词面几乎不重叠。
       → 关键词检索在这个鸿沟上的天花板在哪？哪部分必须交给 Stage 2 的向量检索？
         别急着说"全都交给向量"，说清楚哪些是关键词能救的。

        -

    3) 现在 golden 只有 1 条查询。你在这 1 条上把分数调到最优，
       和你在 20 条上表现稳定，是两件完全不同的事。
       → 前者叫过拟合测试集。你打算怎么扩充 golden，才能证明 V4 不是碰巧？

    实现约束：
      - 只准用 problem_model 里已有的字段，不准硬编码 "报价" "比价" 等本例答案词
      - 写完必须自己跑通，把输出贴在 REFLECTION 里
    """
    raise NotImplementedError("V4 由你实现 —— 这是本圈的训练点")


VARIANTS = [
    ("V0 全字段拼接（你的 build_query）", v0_all_fields),
    ("V1 剔除 unknown 噪声", v1_drop_unknown),
    ("V2 只留 explicit + 高置信度", v2_high_confidence_only),
    ("V3 极简：scenario + function", v3_scenario_and_function),
    ("V4 你的策略", v4_your_strategy),   # 写好后取消注释
]


def main() -> None:
    docs = R.load_corpus()
    golden = json.loads(R.GOLDEN_FILE.read_text("utf-8"))
    if not docs or not golden:
        print("⚠️  语料或 golden 为空")
        return

    k = R.TOP_K
    print("=" * 78)
    print(f"消融实验 | 语料 {len(docs)} 篇 | golden {len(golden)} 条 | k={k}")
    print("=" * 78)
    print(f"{'策略':<34}{'recall':>8}{'precision':>11}{'MRR':>8}")
    print("-" * 78)

    positives = set()
    for case in golden:
        positives.update(case["relevant_doc_ids"])

    for name, fn in VARIANTS:
        try:
            rep = R.evaluate(R.KeywordRetriever(docs), golden, k=k, n_docs=len(docs), query_builder=fn)
        except NotImplementedError as exc:
            print(f"{name:<34}   ⏭️  跳过（{exc}）")
            continue
        print(f"{name:<34}{rep['mean_recall_at_k']:>8}{rep['mean_precision_at_k']:>11}{rep['mrr']:>8}")
        for row in rep["per_query"]:
            got = row["got_top_k"]
            # 召回但 golden 未标注的文档：如果它语义上其实是正例，precision 就是"假低"
            unlabeled = [d for d in got if d not in positives]
            tag = ""
            if unlabeled:
                roles = {d: next((x.metadata.get("role", "?") for x in docs if x.doc_id == d), "?") for d in unlabeled}
                sem_pos = [d for d, r in roles.items() if "正例" in r]
                tag = f"  ⚠️ 召回但 golden 未标: {unlabeled}"
                if sem_pos:
                    tag += f"\n{'':<34}     ↳ 其中语义上是正例的: {sem_pos} → precision 被低估了！"
                else:
                    tag += f"\n{'':<34}     ↳ 真实干扰: {roles}"
            print(f"{'':<34}  {row['query_id']}: 期望={row['expected']} 实际={got}{tag}")

    print("-" * 78)
    print("怎么读这张表：")
    print(f"  · precision@{k} 的分母恒为 k，而 golden 只标了 {len(positives)} 个正例，")
    print(f"    所以 precision 的理论上限是 {len(positives)}/{k} = {len(positives)/k:.4f}。")
    print("    四个变体都顶到这个上限 → 指标饱和，仍然测不出策略差异。")
    print("  · 但注意 got_top_k 的内容已经不一样了：query 构造确实开始影响检索结果，")
    print("    只是现有 golden 无法为这种影响打分。")
    print("  · 结论：现在缺的不是语料规模，而是 golden 的正例标注覆盖率。")
    print("    → 这正是 Step 7（多正例 golden）要解决的，也是本圈第二个训练点。")
    print("=" * 78)


if __name__ == "__main__":
    main()
