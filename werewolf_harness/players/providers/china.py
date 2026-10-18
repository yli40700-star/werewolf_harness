"""The six Chinese providers, as data.

Each one speaks OpenAI's chat-completions shape, and each one deviates from it
somewhere. The deviations live here as fields on a row rather than as branches
in the client, so adding a seventh provider is a table entry and nothing else.

Researched 2026-09-03 against live documentation. Two kinds of claim are mixed
in here and they are NOT the same:

  base_url   probed with a real request -- every one below answered a structured
             401, which proves the endpoint exists and parses `Bearer`.
  the rest   read off a documentation page. Model ids especially go stale: every
             one of these vendors turned over a generation in 2026, and the ids
             that were current a year ago are gone from the docs.

Nothing here has been run against a paid key yet, so no row claims a measured
latency. `cli providers` probes whatever keys are present and prints what it
finds; that is what turns a row from documentation into fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    """One vendor's endpoint and the paperwork to get a key for it."""

    key: str
    label: str                      # what a person calls it
    base_url: str                   # the OpenAI-compatible endpoint
    env_key: str                    # which variable in .env holds the token
    console: str                    # where to sign up and get the key
    topup: str = ""                 # where to add money, when it is a distinct page
    note: str = ""                  # what bites, in one line
    free: str = ""                  # what an unpaid account can actually do here


PROVIDERS: dict[str, Provider] = {
    "qwen": Provider(
        "qwen", "通义千问 / 阿里云百炼",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "LLM_KEY_QWEN",
        "https://bailian.console.aliyun.com/?tab=model#/api-key",
        note="文档在推业务空间专属域名 {WorkspaceId}.cn-beijing.maas.aliyuncs.com，"
             "旧通用域名仍可用且不需要 WorkspaceId；停用日期未公布",
        free="每个模型独立约 100 万 token / 90 天，领取和使用都不用实名，但只在华北2（北京）有效；限流不分免费付费（qwen-plus 30000 RPM），单局约 30 万 token，够跑约 3 局/模型",
    ),
    "doubao": Provider(
        "doubao", "豆包 / 火山方舟",
        "https://ark.cn-beijing.volces.com/api/v3",
        "LLM_KEY_DOUBAO",
        "https://console.volcengine.com/ark/region:cn-beijing/apikey",
        note="用 API Key 时 model 填真实模型名；只有走 AK/SK 签名才需要 ep-xxxx 接入点 id",
        free="注册即得每模型独立的免费额度，但官方不公示数值；协作奖励计划要开后付费并授权采集对话，且 2026-09-01 起 31 款模型已退出",
    ),
    "hunyuan": Provider(
        "hunyuan", "腾讯混元",
        "https://api.hunyuan.cloud.tencent.com/v1",
        "LLM_KEY_HUNYUAN",
        "https://console.cloud.tencent.com/hunyuan/start",
        note="别接 hunyuan.tencentcloudapi.com，那套要 TC3-HMAC-SHA256 签名，不是这个",
        free="10 款模型共享 100 万 token / 1 年，约够 3 局；默认并发 5；tools 只在 turbos / t1 / functioncall 上生效，平台正在迁往 TokenHub",
    ),
    "zhipu": Provider(
        "zhipu", "智谱 GLM",
        "https://open.bigmodel.cn/api/paas/v4",
        "LLM_KEY_ZHIPU",
        "https://open.bigmodel.cn/",
        note="Bearer 直接放 API Key，老的 JWT 签名已非必需",
        free="glm-4.7-flash 名义上永久免费且支持工具调用，但 2026-09-03 实测 16 次里 15 次 429（共享算力限流），"
             "当天等于不可用；付费的 glm-5.3 正常。官方 FAQ 说调 API 不强制实名",
    ),
    "deepseek": Provider(
        "deepseek", "DeepSeek",
        "https://api.deepseek.com",
        "LLM_KEY_DEEPSEEK",
        "https://platform.deepseek.com",
        note="带不带 /v1 都可以；高峰时段（工作日 9-12、14-18）价格翻倍",
        free="没有免费额度，余额为 0 直接 402。要用必须先充值——网上说的注册送 token 无官方来源",
    ),
    "kimi": Provider(
        "kimi", "Kimi / 月之暗面",
        "https://api.moonshot.cn/v1",
        "LLM_KEY_KIMI",
        "https://platform.kimi.com/console/account",
        "https://platform.kimi.com/console/pay",
        note="控制台迁到了 platform.kimi.com，但 API 域名仍是 api.moonshot.cn（api.kimi.com 是 404）；"
             "Tier 0 并发 1 / RPM 3，不充值跑不动 agent",
        free="认证送 ￥15 券但不能用于旗舰 k3；不充值停在 Tier 0：并发 1 / RPM 3，等于每次调用间隔 20 秒，一局至少 20 分钟。充 ￥50 进 Tier 1 才可用",
    ),
}


@dataclass(frozen=True)
class Candidate:
    """One model on one provider, with the quirks needed to call it."""

    model: str
    provider: str
    tier: str                       # flagship | cheap
    tool_choice: str = "required"   # lowered where the vendor is known not to take it
    retired: str = ""               # why it is off the menu, if it is
    extra_body: dict = field(default_factory=dict)
    timeout_s: float = 120.0
    temperature: float | None = None  # pinned where the vendor allows exactly one value
    note: str = ""
    verified: bool = False          # True only once a real call has been made
    measured: str = ""              # what that call showed; empty until verified


# Ordered cheap-first inside each provider: a table of eight AI seats is run on
# the cheap tier, and the flagship is what you put in one or two seats to see
# whether the extra money buys better play.
CANDIDATES: list[Candidate] = [
    # -- DeepSeek. Probed 2026-09-03 on a real 12p vote turn (13 tools, 6.2K chars).
    # Thinking mode refuses tool_choice="required" outright ("Thinking mode does not
    # support this tool_choice"), and with `auto` its thinking tokens eat the output
    # budget: 1 in 8 calls ended finish_reason=length before emitting the call, at
    # 3-18s. Thinking off: 8/8 at 0.6-3.0s. The ReAct loop already makes the model
    # reason out loud step by step, so the built-in thinking is buying a slower,
    # less reliable version of something we already have.
    Candidate("deepseek-v4-flash", "deepseek", "cheap",
              extra_body={"thinking": {"type": "disabled"}}, timeout_s=90.0, verified=True,
              measured="真实投票回合 8/8，0.6-3.0s",
              note="思考模式不接受 tool_choice=required；strict 模式要换 base_url 到 /beta"),
    Candidate("deepseek-v4-pro", "deepseek", "flagship",
              extra_body={"thinking": {"type": "disabled"}}, timeout_s=90.0, verified=True,
              measured="单轮工具调用 1.2s",
              note="同上：思考关掉才能用 required"),

    # -- Qwen
    Candidate("qwen3.8-flash", "qwen", "cheap",
              extra_body={"parallel_tool_calls": True},
              note="总结那一轮必须把 tool_choice 去掉，否则会一直返回工具调用，容易死循环"),
    Candidate("qwen3.8-max", "qwen", "flagship", extra_body={"parallel_tool_calls": True}),

    # -- Doubao
    Candidate("doubao-seed-2-1-turbo-260628", "doubao", "cheap",
              note='tool_choice="required" 是否支持未验证，先按 required 打，被拒就降 auto'),
    Candidate("doubao-seed-2-1-pro-260628", "doubao", "flagship"),

    # -- Kimi
    # Probed 2026-09-03. Kimi allows exactly ONE temperature, and which one depends
    # on thinking: 1 with thinking on, 0.6 with it off -- any other value is a 400.
    # k2.6 additionally refuses tool_choice="required" while thinking ("'required'
    # is incompatible with thinking"), so thinking off is the only配置 that gives
    # us the one-call-per-step contract on both models.
    Candidate("kimi-k2.6", "kimi", "cheap", temperature=0.6,
              extra_body={"thinking": {"type": "disabled"}}, verified=True,
              measured="单轮工具调用 1.5s",
              note="思考开着时 temperature 只能 1，且不接受 tool_choice=required"),
    Candidate("kimi-k3", "kimi", "flagship", temperature=0.6,
              extra_body={"thinking": {"type": "disabled"}}, verified=True,
              measured="单轮工具调用 3.1s",
              note="思考开着也能用 required（与 k2.6 不同），但同样只认 temperature=1"),

    # -- GLM. Probed 2026-09-03. The free model is free and unusable: 15 of 16 calls
    # came back 429 "该模型当前访问量过大" whether or not tool_stream was set. It is
    # shared capacity, not a quota, so waiting does not help and retrying is what
    # everyone else is already doing. Left retired rather than deleted: it costs
    # nothing to try again another day.
    Candidate("glm-4.7-flash", "zhipu", "cheap", tool_choice="auto",
              extra_body={"tool_stream": True}, verified=True,
              retired="免费共享算力常年 429，实测 16 次只通 1 次",
              measured="16 次里 15 次 429，实际打不通",
              note="永久免费但共享算力，长期限流；能通的时候再考虑"),
    # tool_stream turned out NOT to be required on a direct connection (it is a
    # Bailian-hosting requirement); kept because it costs nothing and the vendor
    # documents a tool-stream mode. `required` works, unlike on Bailian's GLM.
    Candidate("glm-5.3", "zhipu", "flagship", timeout_s=90.0, verified=True,
              extra_body={"tool_stream": True, "thinking": {"type": "enabled"},
                          "reasoning_effort": "low"},
              measured="单轮工具调用 4/4，中位 4.2s",
              note="始终思考，传 thinking=disabled 直接 400；默认 reasoning_effort=max 又贵又慢，压到 low"),

    # -- Hunyuan: the least verified. Model ids are a guess until a probe says otherwise.
    Candidate("hunyuan-turbos-latest", "hunyuan", "cheap", tool_choice="auto",
              note="模型 id 与工具调用支持均未验证，等 cli providers 探活"),
]

CANDIDATE_BY_MODEL: dict[str, Candidate] = {c.model: c for c in CANDIDATES}


def provider_of(model: str) -> Provider | None:
    """The vendor a model belongs to, or None when it is not one of these."""
    c = CANDIDATE_BY_MODEL.get(model)
    return PROVIDERS[c.provider] if c else None
