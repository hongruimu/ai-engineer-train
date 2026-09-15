# Circle 1 引导手册

你说过「思路混乱、不要完全让我自己想」。这份文档就是为此写的：**我给结构和骨架，你只填判断和实现。**

---

## 一、这一圈到底在练什么

你的代码核实结果是：**验证层很强，生成层为零。**

- 你做过：pixelmatch 像素比对、8 个 phase gate 阈值判定、C1-C8 lint 门禁、49 个实质测试
- 你没做过：用代码调一次 LLM、解析一次结构化输出、写一个代码级循环

Circle 1 只补后面那一半。产品建模、Capability Graph、RAG、UI 全部挂起。

**一句话目标**：让一段客户原话，经过一次真实的 LLM 调用，变成一份能通过 pydantic 校验的结构化 ProblemModel。

---

## 二、文件清单与分工

| 文件 | 状态 | 你要做什么 |
| --- | --- | --- |
| `schema.py` | ✅ 我已写好，能跑 | **读懂**它，尤其是三个 `@model_validator`。觉得字段不合采购场景就改，但要能说出为什么 |
| `mock_example.json` | ✅ 正例，校验通过 | **对照看**：证据分级、`unknown` 约束怎么写 |
| `mock_mistake.json` | ✅ 反例，必须被拦下 | **看报错**：它复刻的就是你上一轮那个错 |
| `check_examples.py` | ✅ 能跑 | 运行它，看正反两面 |
| `extract.py` | ⚠️ 骨架，含 TODO | **你的主要工作**：填 `build_prompt()` 和 `call_llm()` |
| `my_problem_model.json` | ❌ 不存在 | **你手写一份**，让它通过校验 |

环境已备好：`.venv` 里装了 pydantic 2.13.5，Python 3.14.6。

---

## 三、按这个顺序做（别跳）

### 第 1 步：跑一遍看现象（10 分钟）

```bash
cd circle-01
./.venv/bin/python check_examples.py
```

你会看到正例通过、反例被拦下。拦下它的报错是：

> 以下期望结果把握太低（confidence<0.5），不应作为客户诉求

**这就是你上一轮犯的错**：把「推荐供应商」写进了「客户核心诉求」，但客户从没要求推荐供应商。在这里它会被代码拦住，而不是等到面试官问你。

### 第 2 步：手写 `my_problem_model.json`（40 分钟）

不要先看 LLM。先自己手写一份，目标是**通过校验**。

这一步逼你想清楚三件事，也正是你上一轮漏掉的：

1. 哪些是客户明说的 → `source: "explicit"` + 必须有 `quote`
2. 哪些是你推断的 → `source: "inferred"` + 必须有 `confidence`
3. 客户没提的约束 → 不能漏，`value` 填 `"unknown"`，`source` 填 `"inferred"`，`confidence` 填 `0.0`

`constraints` 必须覆盖这四类：`deployment`（私有化还是云）、`data_sensitivity`（报价单是否商业机密）、`volume`（日均多少份）、`integration`（要不要对接 ERP）。少一类就抛错。

验证：

```bash
./.venv/bin/python -c "
import json, pathlib
from schema import ProblemModel
p = pathlib.Path('my_problem_model.json')
m = ProblemModel.model_validate(json.loads(p.read_text('utf-8')))
print('✅ 通过')
"
```

**为什么先手写**：你手写时会被 validator 逼着做证据分级，这个体感比看十遍讲解都有用。等你自己写通过了，再去调 LLM，你就知道 prompt 里该要求什么。

### 第 3 步：填 `extract.py` 的 `build_prompt()`（40 分钟）

把你在第 2 步被 validator 教育过的规则，写进 system prompt：

- 输出严格 JSON，字段清单是什么
- explicit 必须附原话 quote，不许编
- inferred 必须附 confidence
- 四类 constraints 不许漏，不知道填 unknown

### 第 4 步：填 `call_llm()`（20 分钟）

需要一个 API key。任选一个 OpenAI 兼容的服务（百炼 DashScope、DeepSeek 都行），`extract.py` 里已经留了 `openai` SDK 的示例代码，取消注释、改 base_url 和 model 名即可。

需要先装 SDK：

```bash
./.venv/bin/pip install openai
```

### 第 5 步：跑出 Evidence（30 分钟）

```bash
export DASHSCOPE_API_KEY=你的key
./.venv/bin/python extract.py
```

要交的 Evidence：

1. **≥3 次运行的 `output.json`** + 每次的校验结果 + token 用量
2. **一次失败案例**：把输入里的「采购人员」改成「相关人员」，看模型会不会开始编造角色。这是故意注入歧义，考的是你的系统会不会把幻觉当事实

---

## 四、验收标准（我按这个判）

| 项 | 标准 |
| --- | --- |
| Goal | 客户原话 → 通过校验的 ProblemModel |
| Criteria | ① 用真实 LLM 调用，不是手写 JSON 冒充 ② 证据分级正确：explicit 都有真实 quote，inferred 都有 confidence ③ 四类 constraints 齐全，未知项是 `unknown` 不是编造 ④ 校验失败会带错误信息重试，最终失败显式抛出 ⑤ 有失败案例记录 |
| Evidence | 3+ 次 output.json、token 用量、歧义注入的失败案例 |
| Evaluation | 我逐条对照 Criteria 检查，并实际运行你的代码 |
| Acceptance | 全部满足才算过。差一项我指出具体哪项、为什么 |

**刻意排除，别碰**：Agent、RAG、LangGraph、UI、部署、Capability Graph、V1/V2 演进。

---

## 五、你上一轮的四个缺陷，分别怎么被这次的结构接住

| 你的缺陷 | 这次的对策 |
| --- | --- |
| 没区分事实与推断（把「推荐供应商」当客户诉求） | `SourcedClaim` + `check_no_inferred_leaked_as_explicit` 直接抛错 |
| Constraints 整类缺失 | `check_minimum_coverage` 强制四类，缺一即抛 |
| 没有字段级 schema，只有散文条目 | `schema.py` 已经是字段级、带类型的 |
| RAG 技术误判（比价场景不需要向量检索） | 本圈刻意排除 RAG。跨供应商比价是有界文档集上的结构化抽取 + 确定性比较，你上一轮自己判对了「比较属确定性工作」，这条判断保留 |

---

## 六、卡住了怎么办

按顺序试：

1. 跑 `check_examples.py`，看正反例的差异
2. 看 validator 抛的具体错误信息 —— 它写得很直白，会告诉你缺哪个类别
3. 把你手写的 `my_problem_model.json` 和 `mock_example.json` 逐字段对比
4. 还卡住就直接把你的 JSON 贴给我，我指出具体哪个字段不对

**不要**：跳过第 2 步直接调 LLM。手写那一步是这一圈真正的训练点。
