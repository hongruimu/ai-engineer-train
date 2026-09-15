"""Circle 1 / Test Point C1-01 —— ProblemModel 契约定义。

这个文件已经写好了，你可以直接用。你的任务不是重写它，而是：
  1. 读懂它，特别是三处 @model_validator —— 那是「证据分级」被强制的地方；
  2. 检查字段是否符合你对采购比价这个领域的理解，觉得缺字段/多字段就改，
     改完要能自己说出为什么改；
  3. 写一份 my_problem_model.json，让它通过校验。

设计意图（面试时你可以这样讲）：
  客户原话里混着「他明说的事实」和「我们推断出来的东西」。如果不在数据结构层面
  把两者分开，下游的 Solution 设计、报价、交付标准全都建立在你分不清真假的输入上。
  所以这里用 source / quote / confidence 三个字段把证据等级固化进 schema，
  再用 validator 在解析时就拦住不合规的输出 —— 而不是等到人工 review 时才发现。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator, ConfigDict


class EvidenceSource(str, Enum):
    """每条陈述的证据来源。这是本次训练最核心的约束。"""

    explicit = "explicit"  # 客户原话里直接说了
    inferred = "inferred"  # 由你/模型推断得出，客户没明说


class SourcedClaim(BaseModel):
    """一条带证据来源的陈述。pain_points 与 desired_outcomes 都用它。

    规则：
      source=explicit → 必须给出 quote（客户原话片段）
      source=inferred → 必须给出 confidence（0~1）
    """

    model_config = ConfigDict(extra="forbid")
    description: str = Field(..., min_length=2)
    source: EvidenceSource
    quote: Optional[str] = Field(
        None, description="explicit 时必填：客户原话中的片段"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="inferred 时必填：0~1"
    )

    @model_validator(mode="after")
    def check_evidence(self) -> "SourcedClaim":
        if self.source is EvidenceSource.explicit and not self.quote:
            raise ValueError(
                f"explicit 的陈述必须给出客户原话片段 quote：{self.description!r}"
            )
        if self.source is EvidenceSource.inferred and self.confidence is None:
            raise ValueError(
                f"inferred 的陈述必须给出 confidence：{self.description!r}"
            )
        return self


class Constraint(BaseModel):
    """一条约束。同样受证据规则约束：explicit 要有 quote，inferred 要有 confidence。

    关键规则：客户没说的约束，不许整条漏掉，也不许把猜测写成事实。
    正确做法是 category 照常列出、value 填 "unknown"、source 填 "inferred"、
    confidence 填 0.0 —— 这样下游知道「这一项待向客户澄清」，
    而不是被一个看起来很确定的假值误导。

    这就是上一轮你整类漏掉 Constraints 的补救：不是靠自觉记住，
    而是 check_minimum_coverage 会在缺类别时直接抛错。
    """

    model_config = ConfigDict(extra="forbid")
    category: str = Field(
        ...,
        description=(
            "必填类别之一："
            "deployment / data_sensitivity / volume / integration / budget / template"
        ),
    )
    value: str = Field(..., description="具体内容；确实不知道就填 'unknown'")
    source: EvidenceSource
    quote: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_evidence(self) -> "Constraint":
        if self.source is EvidenceSource.explicit and not self.quote:
            raise ValueError(
                f"explicit 的约束必须给出客户原话片段 quote：{self.category}"
            )
        if self.source is EvidenceSource.inferred and self.confidence is None:
            raise ValueError(f"inferred 的约束必须给出 confidence：{self.category}")
        return self


class ProblemModel(BaseModel):
    """客户问题的结构化模型。这是产品的第一个核心资产。"""

    # 存在ProblemModel如果未定义相关额外字段会报错，例如：第114行的额外字段如果不定义，在做校验时会报错
    model_config = ConfigDict(extra="forbid")
    customer_raw_input: str = Field(..., description="客户原话，原样保存")

    # --- 领域定位 ---
    industry: str = Field(..., description="行业")
    business_function: str = Field(..., description="业务职能，如 采购")
    scenario: str = Field(..., description="具体场景")

    # --- 痛点与期望结果：必须带证据等级 ---
    pain_points: list[SourcedClaim]
    desired_outcomes: list[SourcedClaim]
    # 太没把握的推断放这里，不许冒充客户诉求
    open_questions: list[SourcedClaim] = Field(default_factory=list)

    # --- 约束：上一轮你整类漏掉了，这里强制 ---
    constraints: list[Constraint]

    # --- 现状 ---
    data_assets: list[str] = Field(..., description="涉及的输入物料，如 Excel 报价单")
    current_workflow: list[str] = Field(..., description="客户当前的做法，按步骤")
    decision_owner: str = Field(..., description="最终决策由谁做")
    automation_boundary: str = Field(
        ..., description="哪些环节可自动化、哪些必须人工。你上一轮这点答得最好，保留"
    )

    @model_validator(mode="after")
    def check_minimum_coverage(self) -> "ProblemModel":
        if len(self.pain_points) < 2:
            raise ValueError("pain_points 至少 2 条")

        required_categories = {
            "deployment",
            "data_sensitivity",
            "volume",
            "integration",
        }
        present = {c.category for c in self.constraints}
        missing = required_categories - present
        if missing:
            raise ValueError(
                f"constraints 必须覆盖这些类别（不知道就填 value='unknown'）："
                f"{sorted(missing)}"
            )
        return self

    @model_validator(mode="after")
    def check_no_inferred_leaked_as_explicit(self) -> "ProblemModel":
        """拦住上一轮那个具体错误：把推断写成客户核心诉求。

        规则：任何 inferred 且 confidence < 0.5 的期望结果，
        都不允许出现在 desired_outcomes 里 —— 太没把握的推断应该进
        open_questions（本模型暂未设该字段，你可以自己加）。
        """
        weak = [
            o.description
            for o in self.desired_outcomes
            if o.source is EvidenceSource.inferred
            and o.confidence is not None
            and o.confidence < 0.5
        ]
        if weak:
            raise ValueError(
                f"以下期望结果把握太低（confidence<0.5），不应作为客户诉求，"
                f"应移入 open_questions：{weak}"
            )
        return self


# --- 自检：直接运行本文件应无异常 ---
if __name__ == "__main__":
    import json
    import pathlib

    here = pathlib.Path(__file__).parent
    sample = here / "mock_example.json"
    raw = json.loads(sample.read_text("utf-8"))
    raw['xxx'] = 111
    if sample.exists():
        try:
            model = ProblemModel.model_validate(raw)
            print("✅ mock_example.json 通过校验")
            print(f"   痛点 {len(model.pain_points)} 条 / 约束 {len(model.constraints)} 条")
        except Exception as e:
            print(str(e))
    else:
        print("（mock_example.json 不存在，跳过自检）")
