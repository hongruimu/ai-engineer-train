"""跑两份样例：一份应通过，一份应被拦下。

用法：
    ./.venv/bin/python check_examples.py

这是给你看「证据分级到底怎么被强制」的最快方式。看完再动手写自己的
my_problem_model.json。
"""

import json
import pathlib

from pydantic import ValidationError

from schema import ProblemModel

HERE = pathlib.Path(__file__).parent


def check(filename: str, should_pass: bool) -> bool:
    path = HERE / filename
    raw = json.loads(path.read_text("utf-8"))
    raw.pop("_为什么这个文件存在", None)

    try:
        model = ProblemModel.model_validate(raw)
    except ValidationError as exc:
        if should_pass:
            print(f"❌ {filename} 本该通过，却失败了：")
            for err in exc.errors():
                print(f"     {'.'.join(str(x) for x in err['loc'])}: {err['msg']}")
            return False
        print(f"✅ {filename} 按预期被拦下（这是好事）：")
        for err in exc.errors():
            print(f"     {'.'.join(str(x) for x in err['loc'])}: {err['msg']}")
        return True

    if not should_pass:
        print(f"❌ {filename} 本该被拦下，却通过了 —— validator 有漏洞")
        return False

    print(f"✅ {filename} 通过校验")
    print(
        f"     痛点 {len(model.pain_points)} 条 / 期望 {len(model.desired_outcomes)} 条"
        f" / 约束 {len(model.constraints)} 条"
    )
    inferred = [
        p for p in model.pain_points if p.source.value == "inferred"
    ] + [o for o in model.desired_outcomes if o.source.value == "inferred"]
    if inferred:
        print(f"     其中 {len(inferred)} 条是推断（带 confidence，不是事实）")
    return True


if __name__ == "__main__":
    print("=" * 60)
    ok1 = check("mock_example.json", should_pass=True)
    print("-" * 60)
    ok2 = check("mock_mistake.json", should_pass=False)
    print("=" * 60)
    print("两份样例都符合预期" if (ok1 and ok2) else "有问题，需要修 schema")
    print("=" * 60)
    ok3 = check("my_problem_model.json", should_pass=True)
    print("我自己写的没问题" if (ok3) else "有问题，我自己写的不符合schema的校验要求")
