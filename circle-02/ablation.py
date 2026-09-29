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

       回答：通过字段选择和字段重复实现结构化加权。scenario 最能描述具体问题，
       可重复 3 次；business_function 用于限定业务领域，可重复 2 次；explicit
       pain_points 和 explicit desired_outcomes 保留 1 次。过滤 unknown、待澄清字段
       和低置信度 open_questions，避免冗长文本引入大量无语义单字。当前 BM25.search
       会逐次计算 query 中的重复 token，因此这种做法可以相对放大高价值词和 bigram。
       但它不能彻底消除 unigram 噪声，彻底解决仍需调整 tokenizer。
    

    2) pain_points 用的是客户口语（"耗时间""遗漏"），C1 用的是方案术语
       （"横向比价""归一化""异常报价标记"）。两者词面几乎不重叠。
       → 关键词检索在这个鸿沟上的天花板在哪？哪部分必须交给 Stage 2 的向量检索？
         别急着说"全都交给向量"，说清楚哪些是关键词能救的。

       回答：BM25 能处理具有相同或近似词面的内容，例如业务名称、实体、型号、编号、
       明确术语和共享 bigram。字段过滤和结构化加权可以改善这些词的排序。但如果客户
       使用「比较多家供应商价格和规格」，方案使用 RFQ、should-cost、TCO、BOM、MOQ
       等行业术语，双方没有共同 token，再高的 BM25 权重也无法恢复语义相关性。
       这部分需要向量检索完成同义表达和客户口语到行业术语的语义映射。最终可用 RRF
       融合 BM25 与向量排名；完整解决、半相关和是否应弃权等细粒度判断，仍可能需要
       cross-encoder 或 LLM 重排。


    3) 当前 15 条 golden 已参与 V0~V3 的失败分析和 V4 设计，不能再作为完全独立的
       测试集。你打算怎么证明 V4 不是碰巧适配这些查询？

       回答：当前 15 条只作为开发集。V4 只能根据 ProblemModel 的字段结构决策，禁止
       硬编码具体领域词；实现完成后先冻结代码，再新增一批未参与设计的测试查询。
       新测试集应覆盖不同领域、语义改写、同词异义、半相关、多正例和零正例，并在
       不查看检索结果的情况下盲标 relevant_doc_ids。冻结后只运行一次，对比 V0~V4
       的 recall、precision、MRR 和逐查询结果。如果根据测试结果继续修改 V4，该批
       数据就转为开发集，必须再准备新的独立测试集。

    
    实现约束：
      - 只准用 problem_model 里已有的字段，不准硬编码 "报价" "比价" 等本例答案词
      - 写完必须自己跑通，把输出贴在 REFLECTION 里
    """
    confidence_threshold = 0.6
    parts: list[str] = []

    def clean_text(value: object) -> str:
        if not isinstance(value, str):
            return ""
        text = value.strip()
        if not text or text.lower().startswith("unknown") or text.startswith("【待澄清】"):
            return ""
        return text

    def add_weighted(value: object, weight: int) -> None:
        text = clean_text(value)
        if text:
            parts.extend([text] * weight)

    add_weighted(pm.get("industry"), 1)
    add_weighted(pm.get("business_function"), 2)
    add_weighted(pm.get("scenario"), 3)

    for item in pm.get("pain_points", []) or []:
        if isinstance(item, dict) and item.get("source") == "explicit":
            add_weighted(item.get("description"), 1)

    for item in pm.get("desired_outcomes", []) or []:
        if not isinstance(item, dict):
            continue
        is_reliable = item.get("source") == "explicit" or (
            item.get("confidence") or 0
        ) >= confidence_threshold
        if is_reliable:
            add_weighted(item.get("description"), 1)

    return " ".join(parts)


VARIANTS = [
    ("V0 全字段拼接（你的 build_query）", v0_all_fields),
    ("V1 剔除 unknown 噪声", v1_drop_unknown),
    ("V2 只留 explicit + 高置信度", v2_high_confidence_only),
    ("V3 极简：scenario + function", v3_scenario_and_function),
    ("V4 结构加权", v4_your_strategy),
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

    reports = []
    for name, fn in VARIANTS:
        try:
            rep = R.evaluate(R.KeywordRetriever(docs), golden, k=k, n_docs=len(docs), query_builder=fn)
        except NotImplementedError as exc:
            print(f"{name:<34}   ⏭️  跳过（{exc}）")
            continue
        reports.append((name, rep))
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
    print("怎么读这张表（⚠️ 2026-09-29 修正旧版两处错误）")
    print("  旧版说「precision 理论上限 = |P|/k = 5/3 = 1.6667」—— 错。")
    print("    precision 不可能 > 1.0；且上限须逐条算，不能拿全查询正例并集除以 k。")
    print("  旧版说「缺的是 golden 正例标注覆盖率 → Step 7 要解决」—— 已过时，Step 7 已完成。")
    print()

    bm = R.KeywordRetriever(docs)
    v0_name, v0_fn = VARIANTS[0]
    print(f"逐条上限 vs 实测（以 {v0_name} 为例）")
    print(f"  {'查询':<6}{'|P|':>4}{'P上限':>7}{'P实测':>7}{'R上限':>7}{'R实测':>7}   诊断")
    p_up = r_up = p_act = r_act = 0.0
    for case in golden:
        P = len(case["relevant_doc_ids"])
        q = v0_fn(case["problem_model"]) if "problem_model" in case else case.get("query_text", "")
        got = bm.retrieve(q, k)
        # 上限公式：零正例时「完美克制」= 满分；有正例时被 k 与 |P| 的大小关系锁死
        pu = 1.0 if P == 0 else min(1.0, P / k)
        ru = 1.0 if P == 0 else min(1.0, k / P)
        pa = R.precision_at_k(got, case["relevant_doc_ids"], k)
        ra = R.recall_at_k(got, case["relevant_doc_ids"], k)
        p_up += pu; r_up += ru; p_act += pa; r_act += ra
        if P == 0:
            diag = "V0 未弃权；需比较其他策略是否也返回非空"
        elif pa >= pu - 1e-9 and ra >= ru - 1e-9:
            diag = "V0 已达当前 k 的指标上限，只可能持平或下降"
        else:
            diag = "V0 未达上限，存在改进空间"
        print(f"  {case['query_id']:<6}{P:>4}{pu:>7.2f}{pa:>7.2f}{ru:>7.2f}{ra:>7.2f}   {diag}")
    n = max(len(golden), 1)
    print(f"  {'mean':<6}{'':>4}{p_up/n:>7.3f}{p_act/n:>7.3f}{r_up/n:>7.3f}{r_act/n:>7.3f}")
    print()
    metric_sets = {
        (rep["mean_recall_at_k"], rep["mean_precision_at_k"], rep["mrr"])
        for _, rep in reports
    }
    if len(metric_sets) > 1:
        print("结论：Step 10 通过——V0~V3 的整体指标已经拉开，装置具备初步判别力。")
    else:
        print("结论：V0~V3 的整体指标仍完全相同，装置暂时没有判别力。")

    if reports:
        base_name, base_report = reports[0]
        base_rows = {row["query_id"]: row for row in base_report["per_query"]}
        print(f"  逐策略与 {base_name} 对比：")
        for name, rep in reports[1:]:
            ranking_changed = []
            metric_changed = []
            for row in rep["per_query"]:
                base = base_rows[row["query_id"]]
                if row["got_top_k"] != base["got_top_k"]:
                    ranking_changed.append(row["query_id"])
                before = (base["recall"], base["precision"], base["rr"])
                after = (row["recall"], row["precision"], row["rr"])
                if before != after:
                    metric_changed.append(row["query_id"])
            print(
                f"    {name}: top-{k} 排名变化 {len(ranking_changed)} 条 {ranking_changed}；"
                f"指标变化 {len(metric_changed)} 条 {metric_changed}"
            )
    print("  注意：候选顺序或文档发生变化，不一定会改变 recall/precision/MRR；")
    print("  只有相关文档的命中数量或首个相关文档位置变化，当前三个指标才会变化。")
    print()

    q1_case = next((c for c in golden if c["relevant_doc_ids"]), None)
    if q1_case and "problem_model" in q1_case:
        rel = q1_case["relevant_doc_ids"]
        print(f"补充观察：放大 k 查看 {q1_case['query_id']} 的行业黑话正例（共 {len(rel)} 个正例）")
        print(f"  {'策略':<32}{'R@3':>6}{'R@4':>6}{'R@5':>6}{'R@6':>6}   C2 首次进入")
        for name, fn in VARIANTS:
            q = fn(q1_case["problem_model"])
            vals, first_c2 = [], "-"
            for kk in (3, 4, 5, 6):
                got = bm.retrieve(q, kk)
                vals.append(R.recall_at_k(got, rel, kk))
                if first_c2 == "-" and "C2" in got:
                    first_c2 = f"k={kk}"
            print(f"  {name:<32}" + "".join(f"{v:>6.2f}" for v in vals) + f"   {first_c2}")
        print()
        print("  C2 是用行业黑话写的正例（RFQ/should-cost/TCO/MOQ），词面与客户原话零重叠。")
        print("  只有 V0/V1 能召回它——靠的是低置信度字段（open_questions/constraints）里的专业词。")
        print("  这就是 Stage 2 向量检索的正当理由：让 V2 那样干净的 query 也能召回 C2，")
        print("  而不必靠往 query 里塞噪声来扩大词汇覆盖面。")
    print("=" * 78)


if __name__ == "__main__":
    main()
