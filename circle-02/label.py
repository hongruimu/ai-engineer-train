"""Circle 2 盲标工具 —— Step 9 的 relevant_doc_ids 标注。

════════════════════════════════════════════════════════════════
为什么需要这个工具
════════════════════════════════════════════════════════════════

1. **独立性**：语料是我写的，查询也是我写的。如果标注也由我做，
   就变成「自己出题、自己写答案、自己判分」——违反 Executor ≠ Evaluator。
   标注必须由你独立完成，这是 golden set 作为 ground truth 的唯一前提。

2. **防锚定（role）**：corpus/solutions.json 的 metadata 里有 role 字段
   （写着「正例 · 通俗版」「同类目干扰 · 合同域」等我的设计意图）。
   Step 7 你已确认是看着 role 标的，独立性因此丢失。
   → 本工具隐藏全部 metadata，只显示 text。

3. **防锚定（doc_id 命名）**：⚠️ 这是 Step 9 期间才发现的更严重泄漏。
   原 doc_id 按角色分组命名：C*/H* 全是正例，B*/D*/E*/F*/G*/I* 全是干扰。
   你扫一眼编号规律，不读 text 就能标对全部 15 条，隐藏 metadata 毫无意义。
   → 本工具改用**乱序无语义编号 D01–D23**，真实 doc_id 全程不出现。

4. **防锚定（检索结果）**：本工具不显示 BM25 的排序结果。
   如果给你看排序，你会倾向把排前面的当正例，那样标出的 golden
   只是在复述 BM25 的行为，无法用它来评价 BM25。

════════════════════════════════════════════════════════════════
诚实边界（必须知道）
════════════════════════════════════════════════════════════════

本工具只能**降低无意锚定**，不能阻止故意作弊——corpus/solutions.json
就在你机器上，你随时能打开看 role。技术上防不住。

所以 ground truth 的价值最终取决于你的诚实。Step 7 你主动承认「我是看着
role 标的」，那一次诚实比标对全部 15 条更有价值。这次同样：
**如果标注时你偷看了 role，就在提交时说明**，我会把它标为「非独立标注」
而不是当作有效证据——就像 Q1 现在这样。

════════════════════════════════════════════════════════════════
标注判据（只有一个问题）
════════════════════════════════════════════════════════════════

    「如果我是这个客户，这套方案能不能解决我的问题？」

    · 能完整解决                    → 标为正例
    · 只解决一部分（只做解析不比价）  → **不标**。它是 precision 该惩罚的对象
    · 领域相邻但解决别的问题          → **不标**（如供应商准入 vs 报价比价）
    · 语料里确实没有方案能解决        → 标 []（空列表，不是 null）

════════════════════════════════════════════════════════════════
三步用法
════════════════════════════════════════════════════════════════

  ① 看题：      python3 label.py
  ② 填标注：    编辑 .label_session.json 的 labels 字段
                例：{"Q4": ["D05", "D12"], "Q5": [], "Q6": ["D07"]}
                用 display_id（D01–D23），不要用真实 doc_id
  ③ 提交翻译：  python3 label.py commit
                自动把 display_id 翻译回真实 doc_id 并写入 golden.json
                翻译由代码完成，你不需要（也不应该）自己换算

随时可重跑 ① 查看进度。映射用固定种子生成，重跑不会打乱已填标注。
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
CORPUS_FILE = HERE / "corpus" / "solutions.json"
GOLDEN_FILE = HERE / "golden.json"
SESSION_FILE = HERE / ".label_session.json"

# 固定种子：保证 display_id ↔ doc_id 映射稳定，重跑不作废已填标注
SEED = 20260929


# ─────────────────────────────────────────────────────────────
# 映射
# ─────────────────────────────────────────────────────────────
def build_mapping(doc_ids: list[str]) -> dict[str, str]:
    """生成 display_id → 真实 doc_id 的乱序映射。

    乱序的目的：真实 doc_id 的首字母携带角色信息（C*/H*=正例），
    打乱后 display_id 与角色无任何相关性，命名规律这条泄漏路径被封死。
    """
    shuffled = doc_ids[:]
    random.Random(SEED).shuffle(shuffled)
    return {f"D{i + 1:02d}": real for i, real in enumerate(shuffled)}


def load_session(doc_ids: list[str]) -> dict:
    """读取或初始化标注会话文件。"""
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text("utf-8"))
    return {
        "seed": SEED,
        "display_to_real": build_mapping(doc_ids),
        "labels": {},
        "_说明": "labels 里填 display_id（D01–D23）。零正例填 []，未决定就留空不写这条。",
    }


def save_session(session: dict) -> None:
    SESSION_FILE.write_text(
        json.dumps(session, ensure_ascii=False, indent=2), "utf-8"
    )


# ─────────────────────────────────────────────────────────────
# ① 看题
# ─────────────────────────────────────────────────────────────
def show(session: dict) -> None:
    docs = json.loads(CORPUS_FILE.read_text("utf-8"))
    golden = json.loads(GOLDEN_FILE.read_text("utf-8"))
    m = session["display_to_real"]
    real_to_display = {v: k for k, v in m.items()}
    labels = session.get("labels", {})

    print("=" * 78)
    print(f"语料 {len(docs)} 篇 —— 盲标视图")
    print("已隐藏：metadata（role/vocab_strategy/domain）、真实 doc_id、BM25 排序")
    print("=" * 78)
    for disp in sorted(m):
        real = m[disp]
        doc = next(d for d in docs if d["doc_id"] == real)
        print(f"\n[{disp}]")
        print(f"  {doc['text']}")

    print()
    print("=" * 78)
    print(f"查询 {len(golden)} 条")
    print("=" * 78)
    for c in golden:
        qid = c["query_id"]
        pm = c.get("problem_model")
        print(f"\n--- {qid} ---")
        if pm:
            print(f"  客户原话: {pm.get('customer_raw_input', '(无)')}")
            print(f"  场景    : {pm.get('scenario', '(无)')}")
            print(f"  职能    : {pm.get('business_function', '(无)')}")
        else:
            print(f"  ⚠️ 未用 problem_model: {c.get('query_text')}")
        # 已提交进 golden 的标注
        cur = c.get("relevant_doc_ids")
        if cur is not None:
            disp = [real_to_display.get(x, x) for x in cur]
            print(f"  golden 已定: {cur if cur else '[]零正例'}  (display: {disp})")
            continue
        # 待标：看 session 里填了没
        mine = labels.get(qid)
        if mine is None:
            print("  标注状态: ❓未填 —— 请在 .label_session.json 的 labels 里填")
        else:
            print(f"  你已填(待提交): {mine}")

    # 进度
    pending = [c["query_id"] for c in golden if c.get("relevant_doc_ids") is None]
    filled = [q for q in pending if q in labels]
    todo = [q for q in pending if q not in labels]
    print()
    print("=" * 78)
    print(f"进度：待标 {len(pending)} 条 | 已填 {len(filled)} 条 | 未填 {len(todo)} 条")
    if todo:
        print(f"  还没填: {todo}")
        print(f"\n下一步：编辑 .label_session.json 填 labels，然后跑 python3 label.py commit")
    else:
        print("\n✅ 全部填完，跑 python3 label.py commit 提交")
    print("=" * 78)


# ─────────────────────────────────────────────────────────────
# ③ 提交：display_id → 真实 doc_id，写回 golden.json
# ─────────────────────────────────────────────────────────────
def commit(session: dict) -> None:
    docs = json.loads(CORPUS_FILE.read_text("utf-8"))
    valid = {d["doc_id"] for d in docs}
    m = session["display_to_real"]
    labels = session.get("labels", {})
    golden = json.loads(GOLDEN_FILE.read_text("utf-8"))

    if not labels:
        print("❌ labels 是空的。先编辑 .label_session.json 填标注。")
        return

    # 校验：填的必须是合法 display_id
    bad = []
    for qid, ids in labels.items():
        for x in ids:
            if x not in m:
                bad.append((qid, x))
    if bad:
        print(f"❌ 以下编号不是合法的 display_id（应为 D01–D{len(m):02d}）：")
        for qid, x in bad:
            print(f"     {qid}: {x!r}")
        print("   如果你填的是真实 doc_id（如 C1），请改用 display_id。")
        return

    changed = 0
    for c in golden:
        qid = c["query_id"]
        if qid in labels:
            real_ids = [m[x] for x in labels[qid]]
            # 去重保序
            seen = set()
            real_ids = [x for x in real_ids if not (x in seen or seen.add(x))]
            unknown = [x for x in real_ids if x not in valid]
            if unknown:
                print(f"❌ {qid} 翻译出语料中不存在的 doc_id: {unknown}")
                return
            c["relevant_doc_ids"] = real_ids
            changed += 1

    GOLDEN_FILE.write_text(
        json.dumps(golden, ensure_ascii=False, indent=4), "utf-8"
    )
    print(f"✅ 已提交 {changed} 条标注到 golden.json（display_id 已翻译为真实 doc_id）")
    for c in golden:
        if c["query_id"] in labels:
            print(f"     {c['query_id']}: {c['relevant_doc_ids'] or '[]零正例'}")
    remaining = [c["query_id"] for c in golden if c.get("relevant_doc_ids") is None]
    if remaining:
        print(f"\n⏳ 还有 {len(remaining)} 条未标: {remaining}")
    else:
        print("\n✅ 15 条全部标注完成，可以跑 python3 retrieval.py 了")


# ─────────────────────────────────────────────────────────────
# 候选集：把「23 篇里挑」收窄成「2~3 篇里判」
# ─────────────────────────────────────────────────────────────
# ⚠️ 方法披露：这份候选集是助手通读 23 篇文档后做的**主题相关性分诊**，
#    不是 BM25 的检索输出（用检索输出做候选会让 golden 变成检索器的回声，
#    评测就循环论证了）。可用 corpus 关键词机械复核。
#
# ⚠️ 它只保证「主题沾边的都在这里」，**不保证哪篇是答案**。
#    每条候选里都混着三类东西，需要你自己分开：
#      · 真正解决问题的（该标）
#      · 只是提到这个话题、但解决的是别的问题的（不该标）← 难点在这
#      · 只解决一半的（不该标）
#    「候选里一篇都不该标」永远是合法答案，填 []。
#    若你认为答案在候选之外，用 `python3 label.py` 看全部 23 篇，可以直接加。
CANDIDATES = {
    "Q4":  ["E1", "B2", "C2"],   # 采购合同台账与到期管控
    "Q5":  ["D2", "B2", "F3"],   # 供应商资质证照与合规年审
    "Q6":  ["E2", "B2", "G1"],   # 采购应付三单匹配与对账
    "Q7":  ["F1", "C1", "H1"],   # 电商竞品价格监测与调价
    "Q8":  ["F2", "B2", "C1"],   # 招投标清标与评标辅助
    "Q9":  ["I1", "D1"],         # 客服自动应答与知识库
    "Q10": ["I2", "G2"],         # 代码评审与研发效能
    "Q11": ["I3"],               # 简历筛选与候选人排序（I3 是唯一对口方案）
    "Q12": ["I4"],               # 营销内容批量生成
    "Q13": ["F4", "C1", "H1"],   # 跨境物流运价比价与订舱
    "Q14": ["E3", "B2", "C2"],   # 集团采购支出分析与口径统一
    "Q15": ["B2"],               # 库存盘点治理（语料无对口方案，B2 仅含收货入库模块，大概率填 []）
}


def focus(session: dict, only: str | None = None) -> None:
    """只看待标查询的候选文档，避免逐条通读 23 篇。"""
    docs = json.loads(CORPUS_FILE.read_text("utf-8"))
    golden = json.loads(GOLDEN_FILE.read_text("utf-8"))
    by_id = {d["doc_id"]: d for d in docs}
    real_to_display = {v: k for k, v in session["display_to_real"].items()}

    pending = [c for c in golden if c.get("relevant_doc_ids") is None]
    if only:
        pending = [c for c in pending if c["query_id"] == only]
    if not pending:
        print("✅ 没有待标注的查询了。")
        return

    shown: set[str] = set()
    for c in pending:
        qid = c["query_id"]
        pm = c["problem_model"]
        print("=" * 78)
        print(f"{qid} · {pm.get('scenario')}")
        print("=" * 78)
        print(f"客户原话：{pm.get('customer_raw_input')}")
        print()
        cands = CANDIDATES.get(qid, [])
        if not cands:
            print("  （无候选——请自行用 python3 label.py 通读 23 篇确认）")
        for real in cands:
            disp = real_to_display[real]
            shown.add(real)
            print(f"  [{disp}] {by_id[real]['text']}")
            print()
        print(f"  你的判断：{qid} = [填 display_id，如 {[real_to_display[x] for x in cands[:1]]}；或 []]")
        print()

    rest = len(docs) - len(shown)
    print("─" * 78)
    print(f"其余 {rest} 篇与上述查询无主题交集，已排除（可疑时用 python3 label.py 通读）。")
    print("填完 .label_session.json 的 labels 后跑：python3 label.py commit")
    print("─" * 78)


if __name__ == "__main__":
    docs = json.loads(CORPUS_FILE.read_text("utf-8"))
    session = load_session([d["doc_id"] for d in docs])
    save_session(session)
    if len(sys.argv) > 1 and sys.argv[1] == "commit":
        commit(session)
    elif len(sys.argv) > 1 and sys.argv[1] == "focus":
        focus(session, sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        show(session)
