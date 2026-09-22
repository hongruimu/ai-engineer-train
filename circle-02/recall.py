"""Circle 2 前置题脚手架 —— 召回率指标（recall metric）。

这一圈的核心问题：
    没有人工标注的 ground truth 时，怎么量化「信息有没有丢」？

我已经搭好的（机械部分，你不用动）：
  - AtomicFact 数据结构
  - GROUND_TRUTH：12 条原子事实（已用真实输出核实可锚定）
  - text_of_field / text_of_all：从输出 JSON 提取文本
  - compute_recall：算召回率 + 列出漏掉的事实
  - probe_fulldoc / probe_painpoints：两个「参考探针」，已实现，给你看后果
  - self_check：用 C 组(已知字段级丢量级) / D 组(已知不丢) 当验证基准

你要填的关键点（标了 TODO）：
  - judge_hit：「命中」到底怎么判定 —— 这是整个指标的灵魂
  - GROUND_TRUTH 的粒度、以及 inferred 事实要不要纳入

运行（只用标准库，不需要 venv）：
    cd circle-02 && python3 recall.py
"""

import json
import pathlib
from dataclasses import dataclass

CIRCLE1 = pathlib.Path(__file__).resolve().parent.parent / "circle-01"

CUSTOMER_RAW_INPUT = (
    "我们公司每天要处理大量供应商报价单，很多是 Excel 和 PDF，"
    "现在采购人员需要手动比较价格和规格，非常耗时间，而且经常出现遗漏。"
)


# ─────────────────────────────────────────────────────────────
# 原子事实清单（ground truth）
# ─────────────────────────────────────────────────────────────
@dataclass
class AtomicFact:
    id: str
    statement: str          # 这条事实说什么
    anchor: str             # 客户原话中对应的逐字片段（用于机械匹配）
    kind: str = "explicit"  # explicit / inferred


# 粒度是判断点：我把「比较价格和规格」拆成 比较/价格/规格 三条，
# 把「每天大量」拆成 频率/量级 两条 —— 你也可以合。
# 拆得越细 → 分母越大 → 对「丢失」越敏感；合得越粗 → 越迟钝。这个权衡归你。
GROUND_TRUTH = [
    AtomicFact("F1",  "处理频率 = 每天",              "每天"),
    AtomicFact("F2",  "处理量级 = 大量（具体值未知）", "大量"),
    AtomicFact("F3",  "处理对象 = 供应商报价单",       "供应商报价单"),
    AtomicFact("F4",  "格式包含 Excel",              "Excel"),
    AtomicFact("F5",  "格式包含 PDF",                "PDF"),
    AtomicFact("F6",  "执行者 = 采购人员",            "采购人员"),
    AtomicFact("F7",  "动作 = 手动（非自动）",         "手动"),
    AtomicFact("F8",  "动作 = 比较",                 "比较"),
    AtomicFact("F9",  "比较维度 = 价格",             "价格"),
    AtomicFact("F10", "比较维度 = 规格",             "规格"),
    AtomicFact("F11", "痛点1 = 耗时间",              "非常耗时间"),
    AtomicFact("F12", "痛点2 = 经常遗漏",            "经常出现遗漏"),
    # ── 判断点（我故意留空，由你决定加不加）──────────────────
    # 下面这条是 inferred：原话「很多是 Excel 和 PDF」蕴含「存在其他格式」，
    # 但它没有干净的逐字 anchor。
    #   问题：inferred 事实要不要纳入召回的 ground truth？
    #   纳入的话 anchor 怎么定？不纳入的话，「漏掉一个合理推断」算不算丢失？
    AtomicFact("F13", "格式不止 Excel/PDF（蕴含）", "很多是", kind="inferred"),
]


# ─────────────────────────────────────────────────────────────
# 机械部分（已搭好，你不用改）
# ─────────────────────────────────────────────────────────────
ALL_FIELDS = [
    "pain_points", "desired_outcomes", "open_questions", "constraints",
    "data_assets", "current_workflow", "decision_owner", "automation_boundary",
    "industry", "business_function", "scenario",
]


def text_of_field(output: dict, field_name: str) -> str:
    """把某个字段的值拍平成一段可搜索的文本。"""
    v = output.get(field_name)
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts = []
        for item in v:
            parts.append(json.dumps(item, ensure_ascii=False)
                         if isinstance(item, dict) else str(item))
        return " ".join(parts)
    return json.dumps(v, ensure_ascii=False)


def text_of_all(output: dict) -> str:
    return " ".join(text_of_field(output, f) for f in ALL_FIELDS)


def compute_recall(output: dict, judge_fn, facts=None):
    facts = facts or GROUND_TRUTH
    hits = [f for f in facts if judge_fn(f, output)]
    misses = [f for f in facts if not judge_fn(f, output)]
    rate = len(hits) / len(facts) if facts else 0.0
    return rate, hits, misses


# ── 两个参考探针：已实现，目的是让你「看见」定义不同 → 结论相反 ──
def probe_fulldoc(fact: AtomicFact, output: dict) -> bool:
    """全文档级：anchor 出现在任何字段里就算命中。"""
    return fact.anchor in text_of_all(output)


def probe_painpoints(fact: AtomicFact, output: dict) -> bool:
    """字段级：anchor 只出现在 pain_points 里才算命中。"""
    return fact.anchor in text_of_field(output, "pain_points")


# 人工标准答案
# 一次高召回输出当基线
# Quote锚定覆盖率
# 多次召回的union全集合
# llm as a judge

# ─────────────────────────────────────────────────────────────
# 【核心关键点 —— 你来填】
# ─────────────────────────────────────────────────────────────
def judge_hit(fact: AtomicFact, output: dict) -> bool:
    """你真正要交的指标。先想清楚三件事，再写代码。

    1) 命中范围 —— 三选一（或自己设计第四种），并说清它在度量什么：
       · 全文档级：F2(量级) 在 C 组算命中（它进了 constraints/volume），
         召回率高，但掩盖了「量级没进 pain_points」这个真实退化。
       · pain_points 字段级：C 组 F2 不命中，能抓到退化；
         但本来就该出现在别的字段的事实会被误判为丢失。
       · quote 锚定：要求某个 explicit 输出的 quote 真的指回 anchor。
         最严格，但只覆盖 explicit，管不了 inferred。
       没有一种天生正确 —— 选一种，并说清它会漏掉什么。
    
       全文档级：
       pain_points:
       quote:

    2) 粒度 —— 用现在的 12 条，还是改 GROUND_TRUTH？为什么？

    3) 假阳性 —— 什么情况下你的指标会说「没丢」但其实丢了？
       （这正是你 Q4 那三个指标的通病：一致性满分 + 静默丢事实。）

    填完后，到 self_check 里取消那行注释，把你的 judge_hit 和两个探针一起跑、对比。
    """
    # raise NotImplementedError("judge_hit 是本圈核心关键点，请你自己实现")
    return fact.anchor in text_of_all(output)
    # return fact.anchor in text_of_field(output, "pain_points")


# ─────────────────────────────────────────────────────────────
# 验证基准（已搭好）
# ─────────────────────────────────────────────────────────────
def self_check():
    C = ["output-20260917-134934.json", "output-20260917-135021.json"]  # 思考关+temp0
    D = ["output-20260917-190902.json", "output-20260917-190910.json"]  # 思考关+默认温度
    print("=" * 66)
    print("验证基准：C组(已知 pain_points 丢量级) vs D组(已知不丢)")
    print("=" * 66)
    for label, files in [("C 思考关+temp0", C), ("D 思考关+默认", D)]:
        print(f"\n【{label}】")
        for fn in files:
            p = CIRCLE1 / fn
            if not p.exists():
                print(f"  [跳过] {fn} 不存在")
                continue
            out = json.loads(p.read_text("utf-8"))
            r_full, _, m_full = compute_recall(out, probe_fulldoc)
            r_pain, _, m_pain = compute_recall(out, probe_painpoints)
            print(f"  {fn}")
            print(f"    全文档级召回 : {r_full:.2f}   漏 {[f.id for f in m_full]}")
            print(f"    pain字段级   : {r_pain:.2f}   漏 {[f.id for f in m_pain]}")
            # ── 你填完 judge_hit 后，取消下面这行，把你的指标加进对比 ──
            r_mine, _, m_mine = compute_recall(out, judge_hit)
            print(f"    我的指标     : {r_mine:.2f}   漏 {[f.id for f in m_mine]}")

    print("\n" + "─" * 66)
    print("看 C 组那两行：全文档级说『没丢』，字段级说『丢了 F1/F2/F3』。")
    print("同一个输出、两个指标、相反结论 —— 这就是为什么指标的定义比数值重要。")
    print("而『全文档级』正是会给你假阳性的那种：它把退化掩盖成满分。")
    print("─" * 66)


if __name__ == "__main__":
    self_check()
