"""把客户原话变成 ProblemModel —— 这是你要写的部分。

我已经把脚手架搭好了：调用、重试、校验的骨架都在下面，标了 TODO 的地方
才是你要填的。你不需要从零想架构。

运行方式：
    export DASHSCOPE_API_KEY=你的key     # 或其他兼容 OpenAI 协议的服务
    ./.venv/bin/python extract.py

设计意图（面试可讲）：
    这一步是「生成层」。你过去两个项目只有验证层（pixelmatch、lint），
    从没自己调过 LLM、没解析过结构化输出。这个文件补的就是那一半。
    LLM 负责语义抽取，pydantic 负责确定性校验 —— 两者之间是 Contract。
    LLM 输出不可信，必须过 schema；过不了就带着错误信息重试有限次。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types import CompletionUsage

from pydantic import ValidationError

from schema import ProblemModel

# ---- 客户原话：固定输入，不要改 ----
CUSTOMER_RAW_INPUT = (
    "我们公司每天要处理大量供应商报价单，很多是 Excel 和 PDF，"
    "现在采购人员需要手动比较价格和规格，非常耗时间，而且经常出现遗漏。"
)

MAX_RETRIES = 3


def build_prompt(raw_input: str, retry_feedback: str | None = None) -> list[ChatCompletionMessageParam]:
    """构造发给 LLM 的消息。

    TODO(你)：这里是本次训练最需要你自己思考的地方。
    要求 LLM 返回严格符合 ProblemModel 的 JSON，并且：
      - 每个 pain_point / desired_outcome / constraint 都要标 source；
      - source=explicit 必须从原话里摘 quote，不许自己编；
      - source=inferred 必须给 confidence；
      - 客户没提的约束类别，value 填 "unknown"，不许漏也不许编。

    提示：把 schema 的字段清单和上面这些规则写进 system prompt；
    如果是重试，把上一次的 ValidationError 错误信息塞进 user 消息，
    让模型针对具体错误修正 —— 这叫 reflexion，比单纯重试有效得多。
    """

    system = """
    【角色】
    你是采购领域的需求分析师，把客户的口语化描述转成结构化需求模型。

    【输出格式】
    只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块。
    字段清单：
        customer_raw_input: string 原话，原样保存
        industry / business_function / scenario: string
        pain_points: [ {description, source, quote?, confidence?} ]
        desired_outcomes: [同上]
        open_questions: [同上]
        constraints: [ {category, value, source, quote?, confidence?} ]
        data_assets: [string]
        current_workflow: [string]
        decision_owner: string
        automation_boundary: string
    source 只能是 "explicit" 或 "inferred"。
    confidence 是数字 0～1，不是字符串。

    【JSON样例】
        {
            "customer_raw_input": "<客户原话，原样复制>",
            "industry": "<行业>",
            "business_function": "<业务职能>",
            "scenario": "<具体场景>",
            "pain_points": [
                {"description": "<客户明说的痛点>", "source": "explicit", "quote": "<原话中逐字存在的片段>"},
                {"description": "<你推断的痛点>", "source": "inferred", "confidence": 0.8}
            ],
            "desired_outcomes": [
                {"description": "<期望结果，confidence 必须 >= 0.5>", "source": "inferred", "confidence": 0.7}
            ],
            "open_questions": [
                {"description": "<把握太低、不敢当客户诉求的推断>", "source": "inferred", "confidence": 0.3}
            ],
            "constraints": [
                {"category": "deployment", "value": "unknown", "source": "inferred", "confidence": 0.0},
                {"category": "data_sensitivity", "value": "<你的判断>", "source": "inferred", "confidence": 0.8},
                {"category": "volume", "value": "<客户原话怎么描述数量的>", "source": "explicit", "quote": "<原话片段>"},
                {"category": "volume", "value": "unknown —— <量级未知，及为何影响架构与成本>", "source": "inferred", "confidence": 0.0},
                {"category": "integration", "value": "unknown", "source": "inferred", "confidence": 0.0}
            ],
            "data_assets": ["<数据资产>", "<含未说明格式的待澄清项>"],
            "current_workflow": [
                "<客户明说的步骤>",
                "【待澄清】<不知道但必须问的>"
            ],
            "decision_owner": "<谁决策；不知道的部分写明 unknown 及原因>",
            "automation_boundary": "【确定性代码】… 【LLM】… 【必须人工】… 【待定】…"
        }


    【规则】
    R1 explicit 必须附 quote，且 quote 必须是原话中逐字存在的片段；摘不出原文就改标 inferred。
    R2 inferred 必须附 confidence；不许出现 "unknown" 以外的猜测性 value 配 explicit。
    R3 constraints 必须覆盖 deployment / data_sensitivity / volume / integration 四类；
        客户没提的类别照常列出，value 填 "unknown"，source 填 inferred，confidence 填 0.0。
    R4 volume 必须拆成两条：一条 explicit 记客户原话，一条 inferred 记量级未知。
        不许把「客户说大量」和「可能几十到几千份」糅进同一条。
    R5 confidence < 0.5 的推断不许进 desired_outcomes，只能进 open_questions。
    R6 automation_boundary 必须区分三个执行主体：
        【确定性代码】/【LLM】/【必须人工】。不许只写「可自动化 vs 必须人工」二分。
    R7 current_workflow 只写客户明说的步骤；不知道的用「【待澄清】」开头单独列一条。

    【反例】
    以下是三种典型错误，你的输出中不得出现：

    错误1 把推断当事实：
        {"description":"系统应自动推荐供应商","source":"explicit","quote":"..."}
        为什么错：原话里没有「推荐」二字，摘不出 quote。应改为 inferred + confidence。

    错误2 事实与推断混装：
        {"category":"volume","value":"客户说大量，日均可能几十到几千份","source":"explicit",
        "quote":"大量供应商报价单"}
        为什么错：quote 是真的，但 value 里「几十到几千份」是编的。必须拆两条。

    错误3 角色漂移：
        把「采购人员比较」写成「采购人员决策选定供应商」。
        为什么错：客户只说了比较，选商决策人是谁从未说明。
""".strip()

    user = f"客户原话：\n{raw_input}\n\n请输出符合要求的 JSON。"
    if retry_feedback:
        user += f"\n\n上一次的输出校验失败，错误如下，请针对性修正：\n{retry_feedback}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(messages: list[ChatCompletionMessageParam], want_json: bool = True) -> tuple[str, CompletionUsage]:
    """调用 LLM。返回 (原始文本, 用量信息)。

    默认用 OpenAI 兼容协议，阿里云百炼/DashScope、DeepSeek 等都支持，
    只需改 base_url 和 model 名。没有 key 时会走 mock 分支，方便你先跑通流程。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        print("⚠️  未检测到 API key，返回 mock 响应（仅用于跑通流程，不是真实抽取）")
        return _mock_response(), CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=messages,
        response_format={"type": "json_object"} if want_json else {"type": "text"},
        temperature=1,
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = str(response.choices[0].message.content)
    usage = response.usage
    if usage is None:
        raise ValueError("response.usage is None")
    return text, usage

def _mock_response() -> str:
    """无 key 时的占位响应。故意留了几处不合规，好让你看到校验拦下它的样子。"""
    return json.dumps(
        {
            "customer_raw_input": CUSTOMER_RAW_INPUT,
            "industry": "制造/贸易",
            "business_function": "采购",
            "scenario": "供应商报价单比价",
            "pain_points": [
                {"description": "手动比价耗时", "source": "explicit", "quote": "非常耗时间"}
            ],
            "desired_outcomes": [],
            "constraints": [],
            "data_assets": ["Excel 报价单", "PDF 报价单"],
            "current_workflow": ["人工打开报价单", "手动比较价格规格"],
            "decision_owner": "采购人员",
            "automation_boundary": "比价可自动化，选商需人工",
        },
        ensure_ascii=False,
    )


def extract() -> ProblemModel:
    feedback: str | None = None
    total_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0}

    for attempt in range(1, MAX_RETRIES + 1):
        messages = build_prompt(CUSTOMER_RAW_INPUT, retry_feedback=feedback)
        text, usage = call_llm(messages)

        for k in ("prompt_tokens", "completion_tokens"):
            value = getattr(usage, k)
            if isinstance(value, int):
                total_usage[k] += value

        print(f"--- 第 {attempt} 次尝试 ---")
        print(f"token 用量：{usage}")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            feedback = f"输出不是合法 JSON：{exc}"
            print(f"❌ {feedback}")
            continue

        try:
            model = ProblemModel.model_validate(data)
        except ValidationError as exc:
            feedback = json.dumps(exc.errors(), ensure_ascii=False, indent=2, default=str)
            print("❌ 校验失败，将带错误信息重试：")
            for err in exc.errors():
                print(f"     {'.'.join(str(x) for x in err['loc'])}: {err['msg']}")
            continue

        print("✅ 校验通过")
        print(f"累计 token：{total_usage}")
        return model

    raise RuntimeError(f"重试 {MAX_RETRIES} 次仍未通过校验。最后一次错误：\n{feedback}")


if __name__ == "__main__":

    from datetime import datetime
    try:
        result = extract()
    except Exception as exc:  # noqa: BLE001
        print(f"\n最终失败（这是预期行为之一，失败必须显式抛出，不许静默）：\n{exc}")
        sys.exit(1)

    output_filename_format = "output-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
    out = pathlib.Path(__file__).parent / output_filename_format
    out.write_text(result.model_dump_json(indent=2), "utf-8")
    print(f"\n已写入 {out}")
