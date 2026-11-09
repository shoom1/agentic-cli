"""Startup planning: which providers a catalog's ``load`` waits on, and which are in use.

``startup.plan`` is pure: each case gives the models the settings name (with what
the catalog knows of each), the computed roles, the providers and what is known of
each provider's models, and checks the plan. The catalog's own tests
(``test_catalog.py``) run the same rules against listings.
"""

from __future__ import annotations

import pytest

from agentic_cli.models import ModelRef, ProviderSpec
from agentic_cli.models.startup import Known, NamedModel, StartupPlan, plan
from tests.models.fakes import providers

VLLM = ProviderSpec(base_url="http://gpu-box:8000/v1")
UNLISTED = ProviderSpec(base_url="http://gpu-box:9000/v1", list_models=False)
ALL_KEYS = {"google": "g", "anthropic": "a", "openai": "o", "openrouter": "r"}


def model(
    spec: str,
    *,
    entry: bool = False,
    reference: bool = False,
    tier: str | None = None,
    described: bool = True,
    not_served: bool = False,
) -> NamedModel:
    provider, _, model_id = spec.partition(":")
    return NamedModel(
        ref=ModelRef(provider, model_id),
        entry=entry,
        reference=reference,
        tier=tier,
        described=described,
        not_served=not_served,
    )


def entry(spec: str, **kwargs) -> NamedModel:
    return model(spec, entry=True, **kwargs)


def ref(spec: str, **kwargs) -> NamedModel:
    return model(spec, reference=True, **kwargs)


LISTING = Known(listing=True)
FAILED = Known(failed=True)
LISTED = Known(listing=True, listed=True)
LISTED_AND_FAILED = Known(failed=True, listed=True)


CASES = {
    # (a) no entry can run: every built-in provider with a key and nothing known.
    "no models: the keyed built-ins": dict(
        named=[], keys=dict(google="g", anthropic="a"),
        in_use=("google", "anthropic"), to_list=("google", "anthropic"),
    ),
    "no models: a cached listing or failure is not waited on": dict(
        named=[], keys=dict(google="g", anthropic="a", openai="o"),
        known={"google": LISTING, "anthropic": FAILED},
        in_use=("google", "anthropic", "openai"), to_list=("openai",),
    ),
    "no models and a described reference: still the keyed built-ins": dict(
        named=[ref("google:gemini-2.5-flash", tier="default")], keys=dict(google="g", anthropic="a"),
        in_use=("google", "anthropic"), to_list=("google", "anthropic"),
    ),
    "entries on providers without a key: none can run": dict(
        named=[entry("openai:gpt-5.5", tier="default")], keys=dict(google="g"),
        in_use=("google",), to_list=("google",),
    ),
    "an entry recorded absent cannot run": dict(
        named=[entry("anthropic:claude-sonnet-3-7", tier="default", not_served=True)],
        keys=dict(google="g", anthropic="a"), known={"anthropic": LISTING},
        in_use=("google", "anthropic"), to_list=("google",),
    ),
    # Described models the settings name: nothing to wait on.
    "described entries: no listing": dict(
        named=[
            entry("anthropic:claude-sonnet-4-6", tier="default"),
            entry("openrouter:qwen/qwen3-235b-a22b"),
        ],
        keys=ALL_KEYS, in_use=("anthropic", "openrouter"), to_list=(),
    ),
    "described entries fill the computed roles: no listing": dict(
        named=[
            entry("anthropic:claude-sonnet-4-6", tier="default"),
            entry("anthropic:claude-haiku-4-5", tier="fast"),
        ],
        roles=("default", "fast"), keys=ALL_KEYS, in_use=("anthropic",), to_list=(),
    ),
    "described references beside the entries: no listing": dict(
        named=[
            entry("anthropic:claude-sonnet-4-6", tier="default"),
            ref("openai:gpt-6.2-terra", tier="default"),
            ref("google:gemini-3.8-flash", tier="default"),
        ],
        roles=("default",), keys=ALL_KEYS,
        in_use=("google", "anthropic", "openai"), to_list=(),
    ),
    # (b) a computed role no entry that can run fills.
    "a computed role no entry is of: the keyed built-ins nothing is known of": dict(
        named=[entry("anthropic:claude-sonnet-4-6", tier="default")],
        roles=("default", "strong"), keys=dict(google="g", anthropic="a"),
        known={"anthropic": LISTING},
        in_use=("google", "anthropic"), to_list=("google",),
    ),
    "a computed role only a configured provider's entry is on": dict(
        named=[entry("openrouter:qwen/qwen3-235b-a22b")],
        roles=("default",), keys=dict(anthropic="a", openrouter="r"),
        in_use=("anthropic", "openrouter"), to_list=("anthropic",),
    ),
    "an entry of the role's tier that is recorded absent fills nothing": dict(
        named=[
            entry("google:gemini-9-flash", tier="default", not_served=True),
            entry("google:gemini-2.0-flash"),
        ],
        roles=("default",), keys=dict(google="g", anthropic="a"), known={"google": LISTED},
        in_use=("google", "anthropic"), to_list=("anthropic",),
    ),
    # (c) a model the settings name that nothing describes.
    "an undescribed reference: its provider": dict(
        named=[
            entry("anthropic:claude-sonnet-4-6", tier="default"),
            ref("openrouter:vendor/model", described=False),
        ],
        keys=ALL_KEYS, in_use=("anthropic", "openrouter"), to_list=("openrouter",),
    ),
    "an undescribed entry: its provider": dict(
        named=[entry("local-vllm:Qwen/Qwen3-32B", described=False)],
        specs={"local-vllm": VLLM}, keys=dict(google="g"),
        in_use=("local-vllm",), to_list=("local-vllm",),
    ),
    "an undescribed model a cached listing lacks: listed again": dict(
        named=[entry("anthropic:claude-sonnet-4-6", tier="default"),
               ref("openrouter:vendor/model", described=False)],
        keys=ALL_KEYS, known={"openrouter": LISTING},
        in_use=("anthropic", "openrouter"), to_list=("openrouter",),
    ),
    "an undescribed model a listing recorded absent: not listed for it": dict(
        named=[entry("anthropic:claude-sonnet-4-6", tier="default"),
               ref("openrouter:vendor/model", described=False, not_served=True)],
        keys=ALL_KEYS, known={"openrouter": LISTING},
        in_use=("anthropic", "openrouter"), to_list=(),
    ),
    "an undescribed model whose provider's cached listing failed: not waited on": dict(
        named=[entry("anthropic:claude-sonnet-4-6", tier="default"),
               ref("openrouter:vendor/model", described=False)],
        keys=ALL_KEYS, known={"openrouter": FAILED},
        in_use=("anthropic", "openrouter"), to_list=(),
    ),
    "an undescribed model on a provider never listed (list_models false)": dict(
        named=[entry("box:any-model", described=False)],
        specs={"box": UNLISTED}, keys={}, in_use=("box",), to_list=(),
    ),
    "an undescribed model on a provider without its key": dict(
        named=[entry("anthropic:claude-sonnet-4-6", tier="default"),
               ref("openrouter:vendor/model", described=False)],
        keys=dict(anthropic="a", openrouter=None),
        in_use=("anthropic", "openrouter"), to_list=(),
    ),
    "a reference on an undefined provider is not in use": dict(
        named=[entry("anthropic:claude-sonnet-4-6", tier="default"),
               ref("together:vendor/model", described=False)],
        keys=dict(anthropic="a"), in_use=("anthropic",), to_list=(),
    ),
    # A provider this start listed is never planned again, whatever it found.
    "listed this start: not again": dict(
        named=[ref("openrouter:vendor/model", described=False)],
        keys=dict(google="g", openrouter="r"),
        known={"google": LISTED_AND_FAILED, "openrouter": LISTED_AND_FAILED},
        in_use=("google", "openrouter"), to_list=(),
    ),
    # A model that is an entry and a reference keeps its provider in use as a reference.
    "an entry that is also a reference, recorded absent: its provider stays in use": dict(
        named=[
            model("anthropic:claude-sonnet-3-7", entry=True, reference=True, tier="default",
                  not_served=True),
            entry("google:gemini-2.5-flash", tier="default"),
        ],
        keys=dict(google="g", anthropic="a"), known={"anthropic": LISTING},
        in_use=("google", "anthropic"), to_list=(),
    ),
    "a reference on a built-in provider without a key is in use, not listed": dict(
        named=[entry("google:gemini-2.5-flash", tier="default"),
               ref("anthropic:claude-unknown", described=False)],
        keys=dict(google="g"), in_use=("google", "anthropic"), to_list=(),
    ),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_the_plan(case):
    result = plan(
        case["named"],
        case.get("roles", ()),
        providers(specs=case.get("specs"), **case["keys"]),
        case.get("known", {}),
    )

    assert result == StartupPlan(in_use=case["in_use"], to_list=case["to_list"])


def test_a_provider_nothing_is_said_of_is_known_to_hold_nothing():
    """``known`` may leave a provider out: nothing is known of it."""
    keyed = providers(google="g")

    assert plan([], (), keyed, {}) == plan([], (), keyed, {"google": Known()})


def test_the_order_is_the_providers_not_the_models():
    named = [
        ref("openrouter:vendor/model", described=False),
        ref("local-vllm:Qwen/Qwen3-32B", described=False),
        entry("openai:gpt-5.5", tier="default"),
    ]
    keyed = providers(specs={"local-vllm": VLLM}, openai="o", openrouter="r")

    result = plan(named, ("default",), keyed, {})

    assert result.in_use == ("openai", "openrouter", "local-vllm")
    assert result.to_list == ("openrouter", "local-vllm")


def test_a_second_round_lists_only_the_built_ins_the_first_one_made_needed():
    """Round one lists Google for its undescribed entry and finds the default-tier
    entry absent: the computed role now needs the built-ins, and only Anthropic, which
    nothing is known of yet, is listed in round two. A third plan lists nothing."""
    keyed = providers(google="g", anthropic="a")
    first = [
        entry("google:gemini-9-flash", tier="default"),
        entry("google:gemini-2.0-flash", described=False),
    ]
    after = [
        entry("google:gemini-9-flash", tier="default", not_served=True),
        entry("google:gemini-2.0-flash"),
    ]

    one = plan(first, ("default",), keyed, {})
    two = plan(after, ("default",), keyed, {"google": LISTED})
    three = plan(after, ("default",), keyed, {"google": LISTED, "anthropic": LISTED})

    assert (one.in_use, one.to_list) == (("google",), ("google",))
    assert (two.in_use, two.to_list) == (("google", "anthropic"), ("anthropic",))
    assert three.to_list == ()
