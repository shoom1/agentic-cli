"""``/models`` shows what the model catalog knows.

It prints each provider's status (credentials, listing state, when it was
fetched, how many models, the last error), then a table of the models of every
provider with credentials: reference, context window, max output, thinking,
tools, cache, where the values came from, and the roles that use the model.
``/models <text>`` filters the table by reference, ignoring case, and
``/models refresh`` lists every provider again now and reports what changed.

It works before the workflow is initialized (from the model cache, and says
so), and it never prints a key. Listings come from ``ProviderAPI``, a fake of
every provider's model list; nothing reaches the network.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from agentic_cli.cli.app import BaseCLIApp
from agentic_cli.cli.builtin_commands import HelpCommand
from agentic_cli.cli.commands import CommandRegistry
from agentic_cli.config import BaseSettings
from agentic_cli.models import ModelCatalog
from agentic_cli.workflow import model_services
from tests.models.fakes import ProviderAPI, recorded

KEYS = {
    "GOOGLE_API_KEY": "g-secret-key-1",
    "ANTHROPIC_API_KEY": "a-secret-key-2",
    "OPENAI_API_KEY": "o-secret-key-3",
    "OPENROUTER_API_KEY": "r-secret-key-4",
}
ENDPOINT_KEY = "v-secret-key-5"
LAB = {
    "lab": {
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "LAB_KEY",
        "models": {"Qwen/Qwen3-32B": {"context_window": 32768, "thinking": "chat_template"}},
    }
}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in (*KEYS, "LAB_KEY"):
        monkeypatch.delenv(variable, raising=False)
    return tmp_path


def keys(monkeypatch, *providers: str) -> None:
    for provider in providers:
        variable = f"{provider.upper()}_API_KEY"
        monkeypatch.setenv(variable, KEYS[variable])


@pytest.fixture
def api(monkeypatch) -> ProviderAPI:
    """Every catalog built for the settings lists ``api``'s answers, and caches them."""
    fake = ProviderAPI()

    def build_catalog(settings, endpoints, *, transport=None):
        return ModelCatalog(
            endpoints=endpoints,
            cache_path=model_services.model_cache_path(settings),
            transport=fake.transport,
        )

    monkeypatch.setattr(model_services, "build_catalog", build_catalog)
    return fake


class Screen:
    """The session, rendering what it is given as the terminal would show it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def add_rich(self, renderable: Any) -> None:
        console = Console(file=io.StringIO(), record=True, width=220)
        console.print(renderable)
        self.calls.append(("rich", console.export_text()))

    def add_message(self, role: str, content: str) -> None:
        self.calls.append(("message", content))

    def add_response(self, content: str, markdown: bool = False) -> None:
        self.calls.append(("response", content))

    def add_warning(self, content: str) -> None:
        self.calls.append(("warning", content))

    def add_error(self, content: str) -> None:
        self.calls.append(("error", content))

    def add_success(self, content: str) -> None:
        self.calls.append(("success", content))

    def of(self, kind: str) -> list[str]:
        return [text for k, text in self.calls if k == kind]

    @property
    def text(self) -> str:
        return "\n".join(text for _, text in self.calls)


class App:
    """What a command sees of the app; ``workflow`` raises until one is ready."""

    def __init__(self, settings: BaseSettings, workflow=None) -> None:
        self.settings = settings
        self._workflow = workflow
        self.session = Screen()

    @property
    def workflow(self):
        if self._workflow is None:
            raise RuntimeError("Workflow not initialized yet")
        return self._workflow


def _registry() -> CommandRegistry:
    """The built-in commands, registered as ``BaseCLIApp`` registers them."""
    app = SimpleNamespace(command_registry=CommandRegistry())
    BaseCLIApp._register_builtin_commands(app)
    return app.command_registry


async def models(app: App, args: str = "") -> Screen:
    command = _registry().get("models")
    assert command is not None, "/models is not a built-in command"
    await command.execute(args, app)
    return app.session


async def _ready(settings: BaseSettings, api: ProviderAPI, model: str) -> SimpleNamespace:
    """A ready workflow's view: its loaded catalog and its default model."""
    catalog = model_services.build_catalog(settings, model_services.endpoints_from_settings(settings))
    await catalog.load()
    return SimpleNamespace(catalog=catalog, model=model)


def _row(screen: Screen, ref: str) -> str:
    rows = [line for line in screen.text.splitlines() if f" {ref} " in f" {line} "]
    assert len(rows) == 1, f"{ref}: {rows}"
    return rows[0]


# --- registration ----------------------------------------------------------------------


async def test_models_is_a_built_in_command_listed_in_help():
    registry = _registry()
    app = SimpleNamespace(command_registry=registry, session=Screen())

    await HelpCommand().execute("", app)

    assert registry.get("models") is not None
    assert "/models" in app.session.text


# --- before the workflow is ready ------------------------------------------------------------


async def test_before_the_workflow_is_ready_it_says_so_and_shows_the_table(monkeypatch):
    """No cache and no listing (the test catalog never lists): the shipped
    table's models of each provider with a key."""
    keys(monkeypatch, "google")

    screen = await models(App(BaseSettings()))

    assert any("not initialized" in message for message in screen.of("message"))
    assert "google:gemini-2.5-flash" in screen.text
    assert "not listed" in _row(screen, "google")
    assert not screen.of("error")


async def test_before_the_workflow_is_ready_it_shows_the_cached_listings(api, monkeypatch, home):
    keys(monkeypatch, "google")
    settings = BaseSettings()
    await _ready(settings, api, "google:gemini-2.5-flash")  # an earlier start wrote the cache
    api.calls.clear()

    screen = await models(App(settings))

    assert api.calls == []  # a fresh cache entry needs no listing
    assert "google:gemini-2.0-flash" in screen.text  # listed, not in the table
    assert "fresh" in _row(screen, "google")


# --- the status rows -----------------------------------------------------------------------


async def test_each_provider_s_status(api, monkeypatch):
    keys(monkeypatch, "google", "openai")
    api.answers["openai"] = 401
    settings = BaseSettings()
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow))

    google = _row(screen, "google")
    assert "fresh" in google and "just now" in google and " 6 " in f" {google} "
    openai = _row(screen, "openai")
    assert "failed" in openai and "HTTP 401" in openai
    assert "no key" in _row(screen, "anthropic")
    assert "no key" in _row(screen, "openrouter")
    assert not any("not initialized" in message for message in screen.of("message"))


# --- the model table -------------------------------------------------------------------------


async def test_the_table_shows_each_model_s_record_and_roles(api, monkeypatch):
    keys(monkeypatch, "google")
    settings = BaseSettings(model_roles={"strong": "google:gemini-2.5-pro"})
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow))

    flash = _row(screen, "google:gemini-2.5-flash")
    assert "1,048,576" in flash and "65,536" in flash  # context window and max output, listed
    assert "budget 4096/12288/24576" in flash
    assert "gemini" in flash  # its cache mechanism
    assert "listing, table" in flash
    assert "default" in flash
    assert "fast" in _row(screen, "google:gemini-2.5-flash-lite")  # computed
    assert "strong" in _row(screen, "google:gemini-2.5-pro")  # mapped
    # A provider without a key has no rows.
    assert "anthropic:" not in screen.text


async def test_an_endpoint_s_declared_models_are_shown(api, monkeypatch):
    monkeypatch.setenv("LAB_KEY", ENDPOINT_KEY)
    settings = BaseSettings(model_endpoints=LAB, default_model="lab:Qwen/Qwen3-32B")
    workflow = await _ready(settings, api, "lab:Qwen/Qwen3-32B")

    screen = await models(App(settings, workflow))

    qwen = _row(screen, "lab:Qwen/Qwen3-32B")
    assert "32,768" in qwen and "on/off" in qwen and "declared" in qwen and "default" in qwen
    assert "endpoint" in _row(screen, "lab")


@pytest.mark.parametrize("text", ["FLASH-LITE", "flash-lite", "Flash-Lite"])
async def test_a_filter_matches_the_reference_ignoring_case(api, monkeypatch, text):
    keys(monkeypatch, "google", "anthropic")
    settings = BaseSettings()
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow), text)

    assert "google:gemini-2.5-flash-lite" in screen.text
    assert "google:gemini-2.5-pro" not in screen.text
    assert "anthropic:claude" not in screen.text


async def test_a_filter_that_matches_nothing_says_so(api, monkeypatch):
    keys(monkeypatch, "google")
    settings = BaseSettings()
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow), "no-such-model")

    assert any("no-such-model" in message for message in screen.of("message"))
    assert "google:gemini" not in screen.text


async def test_openrouter_s_models_are_all_shown_and_filterable(api, monkeypatch):
    keys(monkeypatch, "openrouter")
    settings = BaseSettings(default_model="openrouter:qwen/qwen3-235b-a22b")
    workflow = await _ready(settings, api, "openrouter:qwen/qwen3-235b-a22b")

    everything = (await models(App(settings, workflow))).text
    llama = (await models(App(settings, workflow), "llama")).text

    assert "openrouter:meta-llama/llama-3.3-70b-instruct" in everything
    assert "openrouter:qwen/qwen3-235b-a22b:free" in everything
    assert "openrouter:meta-llama/llama-3.3-70b-instruct" in llama
    assert "openrouter:qwen" not in llama


# --- /models refresh -----------------------------------------------------------------------


async def test_refresh_lists_again_and_reports_the_counts_that_changed(api, monkeypatch):
    keys(monkeypatch, "google", "anthropic", "openai")
    google = recorded("google")
    api.answers["google"] = {"models": google["models"][:2]}  # 2 models at startup
    settings = BaseSettings()
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")
    api.answers["google"] = google  # the account now has more
    api.answers["openai"] = 500  # and OpenAI's listing fails
    api.calls.clear()

    screen = await models(App(settings, workflow), "refresh")

    assert sorted(api.calls) == ["anthropic", "google", "openai"]  # every keyed provider, awaited
    report = "\n".join(screen.of("success") + screen.of("message") + screen.of("warning"))
    assert "google: 2 → 6 models" in report
    assert "gemini-2.5-flash-lite" in report  # what it added
    assert "anthropic" not in report  # unchanged
    assert any("openai" in w and "HTTP 500" in w for w in screen.of("warning"))
    assert workflow.catalog.records("google")[2].model  # the live catalog has the new listing


async def test_refresh_with_nothing_new_says_so(api, monkeypatch):
    keys(monkeypatch, "google")
    settings = BaseSettings()
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow), "refresh")

    assert any("unchanged" in message for message in screen.of("message"))
    assert not screen.of("warning")


async def test_refresh_before_the_workflow_is_ready_updates_the_cache(api, monkeypatch, home):
    keys(monkeypatch, "google")
    settings = BaseSettings()

    screen = await models(App(settings), "refresh")

    assert any("not initialized" in message for message in screen.of("message"))
    assert "google" in api.calls
    cache = json.loads(model_services.model_cache_path(settings).read_text())
    assert cache["providers"]["google"]["ok"] is True


# --- keys --------------------------------------------------------------------------------------


async def test_it_never_prints_a_key(api, monkeypatch):
    keys(monkeypatch, *(variable.split("_")[0].lower() for variable in KEYS))
    monkeypatch.setenv("LAB_KEY", ENDPOINT_KEY)
    api.answers["anthropic"] = 401
    api.answers["endpoint"] = 403
    settings = BaseSettings(model_endpoints=LAB)
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    printed = []
    for args in ("", "refresh", "qwen"):
        printed.append((await models(App(settings, workflow), args)).text)
    printed.append((await models(App(settings))).text)  # before the workflow is ready

    output = "\n".join(printed)
    assert "HTTP 401" in output and "HTTP 403" in output  # the failures are shown
    for secret in (*KEYS.values(), ENDPOINT_KEY):
        assert secret not in output
