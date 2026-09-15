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

from pydantic import ValidationError

from schema import ProblemModel

# ---- 客户原话：固定输入，不要改 ----
CUSTOMER_RAW_INPUT = (
    "我们公司每天要处理大量供应商报价单，很多是 Excel 和 PDF，"
    "现在采购人员需要手动比较价格和规格，非常耗时间，而且经常出现遗漏。"
)

MAX_RETRIES = 3


def build_prompt(raw_input: str, retry_feedback: str | None = None) -> list[dict]:
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
    system = """TODO: 写你的 system prompt。
要点：你是采购领域的需求分析师；输出必须是严格 JSON；
每条陈述必须标注 source(explicit/inferred)；explicit 必须附客户原话 quote；
inferred 必须附 confidence(0~1)；constraints 必须覆盖
deployment/data_sensitivity/volume/integration 四类，不知道填 unknown。"""

    user = f"客户原话：\n{raw_input}\n\n请输出符合要求的 JSON。"
    if retry_feedback:
        user += f"\n\n上一次的输出校验失败，错误如下，请针对性修正：\n{retry_feedback}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(messages: list[dict], want_json: bool = True) -> tuple[str, dict]:
    """调用 LLM。返回 (原始文本, 用量信息)。

    默认用 OpenAI 兼容协议，阿里云百炼/DashScope、DeepSeek 等都支持，
    只需改 base_url 和 model 名。没有 key 时会走 mock 分支，方便你先跑通流程。
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("⚠️  未检测到 API key，返回 mock 响应（仅用于跑通流程，不是真实抽取）")
        return _mock_response(), {"prompt_tokens": 0, "completion_tokens": 0, "mock": True}

    # TODO(你)：真正调用。用 openai SDK 或 httpx 都行。示例（需 pip install openai）：
    #
    # from openai import OpenAI
    # client = OpenAI(
    #     api_key=api_key,
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    # resp = client.chat.completions.create(
    #     model="qwen-plus",
    #     messages=messages,
    #     response_format={"type": "json_object"} if want_json else None,
    # )
    # text = resp.choices[0].message.content
    # usage = resp.usage.model_dump()
    # return text, usage
    raise NotImplementedError("请在这里填入真实的 LLM 调用")


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
            if isinstance(usage.get(k), int):
                total_usage[k] += usage[k]

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
    try:
        result = extract()
    except Exception as exc:  # noqa: BLE001
        print(f"\n最终失败（这是预期行为之一，失败必须显式抛出，不许静默）：\n{exc}")
        sys.exit(1)

    out = pathlib.Path(__file__).parent / "output.json"
    out.write_text(result.model_dump_json(indent=2), "utf-8")
    print(f"\n已写入 {out}")
