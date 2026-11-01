"""``/models`` shows the list, finds a model in a full listing, and lists again.

``/models`` shows the list the model pickers offer (``ModelCatalog.offered``):
each model's context window, max output, thinking, tools, cache, where the
values came from, and the roles that use it; then each ``models`` entry the
list leaves out, its Status saying why. Below, each provider in use: its
credentials, listing state, when it was fetched, how many models it lists and
the last error. ``/models <text>`` searches the full listings of every
provider with a key, ignoring case (a provider not in use is listed the first
time), and marks each match the list already has. ``/models refresh`` lists
the providers in use, and those searched, again now and reports what changed.

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
from agentic_cli.workflow import model_services
from agentic_cli.workflow.config import AgentConfig
from tests.conftest import patch_catalogs
from tests.models.fakes import ProviderAPI, openrouter_spec, recorded

KEYS = {
    "GOOGLE_API_KEY": "g-secret-key-1",
    "ANTHROPIC_API_KEY": "a-secret-key-2",
    "OPENAI_API_KEY": "o-secret-key-3",
    "OPENROUTER_API_KEY": "r-secret-key-4",
}
ENDPOINT_KEY = "v-secret-key-5"
LAB = {"lab": {"base_url": "http://localhost:8000/v1", "api_key_env": "LAB_KEY"}}
OPENROUTER = {"openrouter": openrouter_spec()}
LAB_MODELS = {"lab:Qwen/Qwen3-32B": {"context_window": 32768, "thinking": "chat_template"}}


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
    patch_catalogs(monkeypatch, transport=fake.transport, cached=True)
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

    def __init__(self, settings: BaseSettings, workflow=None, agent_configs=()) -> None:
        self.settings = settings
        self._workflow = workflow
        self.agent_configs = tuple(agent_configs)
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
    catalog = model_services.build_catalog(settings, model_services.providers_from_settings(settings))
    await catalog.load()
    return SimpleNamespace(catalog=catalog, model=model)


def _row(screen: Screen, ref: str) -> str:
    rows = [line for line in screen.text.splitlines() if f" {ref} " in f" {line} "]
    assert len(rows) == 1, f"{ref}: {rows}"
    return rows[0]


def _provider_row(screen: Screen, name: str) -> str:
    """``name``'s row in the last Providers table (a model row may name its cache)."""
    providers = [text for text in screen.of("rich") if text.strip().startswith("Providers")]
    rows = [line for line in providers[-1].splitlines() if line.startswith(f"│ {name} ")]
    assert len(rows) == 1, f"{name}: {rows}"
    return rows[0]


def _tables(screen: Screen) -> list[str]:
    """Each table's title, in the order they were shown."""
    return [text.strip().splitlines()[0].strip() for text in screen.of("rich")]


# --- registration ----------------------------------------------------------------------


async def test_models_is_a_built_in_command_listed_in_help():
    registry = _registry()
    app = SimpleNamespace(command_registry=registry, session=Screen())

    await HelpCommand().execute("", app)

    assert registry.get("models") is not None
    assert "/models" in app.session.text


# --- before the workflow is ready ------------------------------------------------------------


async def test_before_the_workflow_is_ready_it_says_so_and_shows_the_list(monkeypatch):
    """No cache and no listing (the test catalog never lists): the list from
    the shipped table."""
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
    # The fast role's model is the listing's newest, not the table's.
    assert "google:gemini-2.5-flash-lite" in screen.text
    assert "google:gemini-3.1-flash-lite-preview" not in screen.text
    assert "fresh" in _row(screen, "google")


async def test_before_the_workflow_is_ready_the_list_has_the_agents_models(monkeypatch):
    keys(monkeypatch, "google", "anthropic")
    settings = BaseSettings(models={"google:gemini-2.5-flash": {}})
    agents = [AgentConfig(name="writer", prompt="Write.", model="anthropic:claude-opus-4-7")]

    screen = await models(App(settings, agent_configs=agents))

    assert "anthropic:claude-opus-4-7" in screen.text
    assert "not listed" in _provider_row(screen, "anthropic")  # in use


def test_the_app_gives_a_command_the_agents_its_workflow_is_built_from():
    from agentic_cli.cli.workflow_controller import WorkflowController

    agents = [AgentConfig(name="writer", prompt="Write.", model="anthropic:claude-opus-4-7")]
    app = BaseCLIApp.__new__(BaseCLIApp)
    app._workflow_controller = WorkflowController(agents, BaseSettings())

    assert app.agent_configs == tuple(agents)


# --- the list --------------------------------------------------------------------------------


async def test_the_list_with_each_model_s_facts_sources_and_roles(api, monkeypatch):
    keys(monkeypatch, "google")
    settings = BaseSettings(
        models={"google:gemini-2.5-flash": {"max_output": 1_000}},
        model_roles={"strong": "google:gemini-2.5-pro"},
    )
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow))

    flash = _row(screen, "google:gemini-2.5-flash")
    assert "1,048,576" in flash and "1,000" in flash  # the listing's window, the entry's max
    assert "budget 4096/12288/24576" in flash
    assert "gemini" in flash  # its cache mechanism
    assert "config, listing, table" in flash
    assert "default" in flash
    assert "strong" in _row(screen, "google:gemini-2.5-pro")  # mapped
    assert "fast" in _row(screen, "google:gemini-2.5-flash-lite")  # computed
    # Listed, but not in the list.
    assert "google:gemini-2.0-flash" not in screen.text
    assert "google:gemini-3-flash-preview" not in screen.text


async def test_an_entry_the_list_leaves_out_says_why(api, monkeypatch):
    keys(monkeypatch, "google")
    settings = BaseSettings(
        model_providers=LAB,
        models={
            "google:gemini-2.5-flash": {},
            "lab:Qwen/Qwen3-32B": {},
            "together:deepseek-ai/DeepSeek-V3.1": {},
            "google:gemini-9-ultra": {},
        },
    )
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow))

    assert "no key: set LAB_KEY" in _row(screen, "lab:Qwen/Qwen3-32B")
    assert "provider 'together' is not defined in model_providers" in _row(
        screen, "together:deepseek-ai/DeepSeek-V3.1"
    )
    assert "not served by google" in _row(screen, "google:gemini-9-ultra")
    flash = _row(screen, "google:gemini-2.5-flash")
    assert "no key" not in flash and "not served" not in flash
    assert _tables(screen)[0] == "Models (3 offered, 3 not)"


async def test_below_the_list_the_providers_in_use(api, monkeypatch):
    keys(monkeypatch, "google", "anthropic")
    settings = BaseSettings(
        models={"google:gemini-2.5-flash": {}}, default_model="google:gemini-2.5-flash"
    )
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow))

    assert _tables(screen) == ["Models (3)", "Providers"]  # the entry, the fast and strong roles'
    assert "fresh" in _row(screen, "google")
    assert " anthropic " not in f" {screen.text} "  # a key, but nothing uses it


async def test_without_a_key_there_are_no_models():
    screen = await models(App(BaseSettings()))

    assert "No models: no provider has a key." in screen.of("message")
    assert _tables(screen) == []


async def test_each_provider_s_status(api, monkeypatch):
    keys(monkeypatch, "google", "openai")
    api.answers["openai"] = 401
    settings = BaseSettings(
        model_providers=OPENROUTER,
        # Roles on providers without a key: in use, so shown.
        model_roles={"strong": "anthropic:claude-opus-4-7", "critic": "openrouter:qwen/qwen3"},
    )
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow))

    google = _row(screen, "google")
    assert "fresh" in google and "just now" in google and " 6 " in f" {google} "
    openai = _row(screen, "openai")
    assert "failed" in openai and "HTTP 401" in openai
    assert "no key" in _row(screen, "anthropic")
    assert "no key" in _row(screen, "openrouter")
    assert not any("not initialized" in message for message in screen.of("message"))


async def test_a_configured_provider_s_models_show_their_entries(api, monkeypatch):
    monkeypatch.setenv("LAB_KEY", ENDPOINT_KEY)
    settings = BaseSettings(
        model_providers=LAB, models=LAB_MODELS, default_model="lab:Qwen/Qwen3-32B"
    )
    workflow = await _ready(settings, api, "lab:Qwen/Qwen3-32B")

    screen = await models(App(settings, workflow))

    qwen = _row(screen, "lab:Qwen/Qwen3-32B")
    assert "32,768" in qwen and "on/off" in qwen and "config" in qwen and "default" in qwen
    assert "│ key " in _row(screen, "lab")


async def test_a_configured_provider_s_credentials(api, monkeypatch):
    """A key, no key (its variable is unset), or none needed (it names none)."""
    keys(monkeypatch, "google")
    providers = {
        **LAB,
        **OPENROUTER,
        "local": {"base_url": "http://localhost:9000/v1"},
    }
    roles = {"critic": "lab:qwen3", "editor": "openrouter:qwen/qwen3", "reader": "local:qwen3"}
    settings = BaseSettings(model_providers=providers, model_roles=roles)
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow))

    assert "│ no key " in _row(screen, "lab")
    assert "│ no key " in _row(screen, "openrouter")
    assert "│ none needed " in _row(screen, "local")


# --- /models <text>: a search of the full listings ----------------------------------------


async def test_a_search_finds_models_in_a_full_listing_and_marks_the_list_s(api, monkeypatch):
    keys(monkeypatch, "google")
    settings = BaseSettings(models={"google:gemini-2.5-flash": {}})
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow), "flash")

    assert _tables(screen) == ["Models matching 'flash' (4)"]
    assert "in the list" in _row(screen, "google:gemini-2.5-flash")
    assert "in the list" in _row(screen, "google:gemini-2.5-flash-lite")  # the fast role's
    assert "in the list" not in _row(screen, "google:gemini-2.0-flash")
    assert "in the list" not in _row(screen, "google:gemini-3-flash-preview")


async def test_a_search_lists_a_provider_not_in_use_the_first_time(api, monkeypatch):
    keys(monkeypatch, "google", "openrouter")
    settings = BaseSettings(model_providers=OPENROUTER, models={"google:gemini-2.5-flash": {}})
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")
    assert api.calls == ["google"]
    api.calls.clear()

    first = await models(App(settings, workflow), "llama")
    again = await models(App(settings, workflow), "llama")
    listed = await models(App(settings, workflow))

    assert api.calls == ["openrouter"]  # then searched in memory
    for screen in (first, again):
        row = _row(screen, "openrouter:meta-llama/llama-3.3-70b-instruct")
        assert "131,072" in row and "in the list" not in row
    assert "fresh" in _row(listed, "openrouter")  # searched: shown with the providers in use
    assert "openrouter:meta-llama" not in listed.text  # but not in the list


@pytest.mark.parametrize("text", ["FLASH-LITE", "flash-lite", "Flash-Lite"])
async def test_a_search_matches_the_reference_ignoring_case(api, monkeypatch, text):
    keys(monkeypatch, "google", "anthropic")
    settings = BaseSettings()
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow), text)

    assert "google:gemini-2.5-flash-lite" in screen.text
    assert "google:gemini-2.5-pro" not in screen.text
    assert "anthropic:claude" not in screen.text


async def test_a_search_that_matches_nothing_says_so(api, monkeypatch):
    keys(monkeypatch, "google")
    settings = BaseSettings()
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow), "no-such-model")

    assert screen.of("message") == ["No model matches 'no-such-model'."]
    assert "google:gemini" not in screen.text


async def test_a_search_finds_what_the_list_leaves_out(api, monkeypatch):
    """OpenRouter's catalog runs to hundreds of models: the list has the one in
    use, and a search finds the others to copy into models."""
    keys(monkeypatch, "openrouter")
    settings = BaseSettings(
        model_providers=OPENROUTER, default_model="openrouter:qwen/qwen3-235b-a22b"
    )
    workflow = await _ready(settings, api, "openrouter:qwen/qwen3-235b-a22b")

    everything = (await models(App(settings, workflow))).text
    llama = (await models(App(settings, workflow), "llama")).text

    assert "openrouter:qwen/qwen3-235b-a22b " in everything
    assert "openrouter:meta-llama/llama-3.3-70b-instruct" not in everything
    assert "openrouter:qwen/qwen3-235b-a22b:free" not in everything
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
    assert _tables(screen) == ["Providers"]


async def test_refresh_with_nothing_new_says_so(api, monkeypatch):
    keys(monkeypatch, "google")
    settings = BaseSettings()
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    screen = await models(App(settings, workflow), "refresh")

    assert any("unchanged" in message for message in screen.of("message"))
    assert not screen.of("warning")


async def test_refresh_never_lists_a_provider_that_is_never_listed(api, monkeypatch):
    keys(monkeypatch, "google")
    settings = BaseSettings(
        model_providers={"local": {"base_url": "http://localhost:9000/v1", "list_models": False}},
        models={"local:qwen3-32b": {}, "google:gemini-2.5-flash": {}},
    )
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")
    api.calls.clear()

    screen = await models(App(settings, workflow), "refresh")

    assert api.calls == ["google"]
    assert screen.of("message") == ["Listed 1 provider(s) again: their models are unchanged."]
    assert "not listed" in _provider_row(screen, "local")


async def test_refresh_lists_the_providers_searched_too(api, monkeypatch):
    keys(monkeypatch, "google", "openrouter")
    settings = BaseSettings(model_providers=OPENROUTER, models={"google:gemini-2.5-flash": {}})
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")
    await models(App(settings, workflow), "qwen")
    api.calls.clear()

    screen = await models(App(settings, workflow), "refresh")

    assert sorted(api.calls) == ["google", "openrouter"]
    assert "fresh" in _row(screen, "openrouter")


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
    settings = BaseSettings(
        model_providers={**LAB, **OPENROUTER}, model_roles={"critic": "lab:Qwen/Qwen3-32B"}
    )
    workflow = await _ready(settings, api, "google:gemini-2.5-flash")

    printed = []
    for args in ("", "refresh", "qwen"):
        printed.append((await models(App(settings, workflow), args)).text)
    printed.append((await models(App(settings))).text)  # before the workflow is ready

    output = "\n".join(printed)
    assert "HTTP 401" in output and "HTTP 403" in output  # the failures are shown
    for secret in (*KEYS.values(), ENDPOINT_KEY):
        assert secret not in output
