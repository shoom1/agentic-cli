"""ADK: what the model runs itself marks the conversation (spec §9.1).

A model-side tool (Google Search, URL context, Vertex AI Search, a built-in or
local code executor) puts search results, pages or program output into the
conversation without a tool call, so its call never passes ``PermissionPlugin``'s
tool callback. The plugin sees the request instead: ADK builds the request's
tools before the model callbacks run.
"""

from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.agents.callback_context import CallbackContext  # noqa: E402
from google.adk.agents.invocation_context import InvocationContext  # noqa: E402
from google.adk.code_executors import BuiltInCodeExecutor, UnsafeLocalCodeExecutor  # noqa: E402
from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.adk.tools import FunctionTool, google_search  # noqa: E402
from google.adk.tools.enterprise_search_tool import EnterpriseWebSearchTool  # noqa: E402
from google.adk.tools.url_context_tool import url_context  # noqa: E402
from google.adk.tools.vertex_ai_search_tool import VertexAiSearchTool  # noqa: E402

from agentic_cli.workflow.adk.permission_plugin import PermissionPlugin  # noqa: E402
from agentic_cli.workflow.config import AgentConfig  # noqa: E402
from agentic_cli.workflow.outside_content import OUTSIDE_CONTENT_KEY, OutsideContentMarks  # noqa: E402
from agentic_cli.workflow.sessions import SessionRef, reset_active_turn, set_active_turn  # noqa: E402
from tests.workflow.test_adk_context_cache import gemini_api, route_gemini  # noqa: E402
from tests.workflow.adk.fakes import FakeAPI  # noqa: E402
from tests.workflow.test_manager_models import KEYS, adk, keys, settings  # noqa: E402

REF = SessionRef(app_name="app", user_id="u", session_id="s")
MODEL = "gemini-2.5-flash"
STORE = "projects/p/locations/global/collections/default_collection/dataStores/d"


@pytest.fixture(autouse=True)
def _hermetic(tmp_path, monkeypatch):
    """A hermetic cwd and HOME, and no provider key unless a test sets one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in KEYS:
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def active_turn():
    token = set_active_turn(REF)
    yield
    reset_active_turn(token)


async def _context(name="root", **agent) -> CallbackContext:
    """A real callback context of a real agent, as ADK passes the plugin's callbacks."""
    service = InMemorySessionService()
    session = await service.create_session(app_name="app", user_id="u", session_id="s")
    agent = LlmAgent(name=name, model=MODEL, **agent)
    return CallbackContext(InvocationContext(
        session_service=service, invocation_id="i", agent=agent, session=session,
    ))


async def _request(*tools) -> LlmRequest:
    """The request ADK builds once each tool has processed it (before the model callbacks)."""
    context = await _context()
    request = LlmRequest(model=MODEL)
    for tool in tools:
        await tool.process_llm_request(tool_context=context, llm_request=request)
    return request


def _builtin_code_execution() -> LlmRequest:
    request = LlmRequest(model=MODEL)
    BuiltInCodeExecutor().process_llm_request(request)
    return request


def _call():
    """A function tool's body."""


@pytest.mark.parametrize("tool, name", [
    (google_search, "google_search"),
    (url_context, "url_context"),
    (EnterpriseWebSearchTool(), "enterprise_web_search"),
    (VertexAiSearchTool(data_store_id=STORE), "retrieval"),
])
async def test_a_model_side_tool_in_the_request_marks_the_conversation(active_turn, tool, name):
    marks = OutsideContentMarks()
    plugin = PermissionPlugin(marks=marks, root="root")
    context = await _context()

    assert await plugin.before_model_callback(
        callback_context=context, llm_request=await _request(tool)
    ) is None
    assert marks.get(REF) == name
    assert context.state.get(OUTSIDE_CONTENT_KEY) == name


async def test_the_built_in_code_executor_s_request_marks_it(active_turn):
    marks = OutsideContentMarks()
    plugin = PermissionPlugin(marks=marks, root="root")
    await plugin.before_model_callback(
        callback_context=await _context(), llm_request=_builtin_code_execution()
    )
    assert marks.get(REF) == "code_execution"


async def test_an_agent_with_a_code_executor_marks_it(active_turn):
    """A local executor runs the model's code blocks and puts their output in
    the conversation; nothing in the request says so."""
    marks = OutsideContentMarks()
    plugin = PermissionPlugin(marks=marks, root="root")
    context = await _context(code_executor=UnsafeLocalCodeExecutor())
    await plugin.before_model_callback(callback_context=context, llm_request=await _request())
    assert marks.get(REF) == "code_executor"
    assert context.state.get(OUTSIDE_CONTENT_KEY) == "code_executor"


async def test_function_tools_and_a_plain_agent_mark_nothing(active_turn):
    marks = OutsideContentMarks()
    plugin = PermissionPlugin(marks=marks, root="root")
    context = await _context()
    await plugin.before_model_callback(
        callback_context=context, llm_request=await _request(FunctionTool(func=_call))
    )
    await plugin.before_model_callback(callback_context=context, llm_request=await _request())
    assert marks.get(REF) is None
    assert OUTSIDE_CONTENT_KEY not in context.state


async def test_the_first_tool_names_the_mark(active_turn):
    marks = OutsideContentMarks()
    plugin = PermissionPlugin(marks=marks, root="root")
    context = await _context()
    await plugin.before_model_callback(
        callback_context=context, llm_request=await _request(url_context, google_search)
    )
    assert marks.get(REF) == "url_context"


async def test_a_sub_agent_s_model_side_tool_marks_and_the_root_s_state_follows(active_turn):
    marks = OutsideContentMarks()
    plugin = PermissionPlugin(marks=marks, root="root")
    sub, root = await _context("researcher"), await _context("root")
    await plugin.before_model_callback(
        callback_context=sub, llm_request=await _request(google_search)
    )
    assert marks.get(REF) == "google_search"
    assert OUTSIDE_CONTENT_KEY not in sub.state
    await plugin.before_model_callback(callback_context=root, llm_request=await _request())
    assert root.state.get(OUTSIDE_CONTENT_KEY) == "google_search"


async def test_a_plugin_without_marks_changes_nothing(active_turn):
    plugin = PermissionPlugin()
    context = await _context()
    await plugin.before_model_callback(
        callback_context=context, llm_request=await _request(google_search)
    )
    assert OUTSIDE_CONTENT_KEY not in context.state


# --- through ADK's own flow ----------------------------------------------------------


def _marked(manager) -> str | None:
    return manager._outside_marks.get(SessionRef(app_name=manager.app_name, user_id="u", session_id="s"))


async def _stored_mark(manager) -> object:
    session = await manager._session_service.get_session(
        app_name=manager.app_name, user_id="u", session_id="s"
    )
    return session.state.get(OUTSIDE_CONTENT_KEY)


async def test_a_config_built_agent_with_google_search_marks_its_conversation(monkeypatch):
    keys(monkeypatch, "google")
    route_gemini(monkeypatch, FakeAPI(gemini_api))
    manager = adk(
        [AgentConfig(name="solo", prompt="Search.", tools=[google_search], include_state_tools=False)],
        settings(default_model=MODEL),
    )
    try:
        [_ async for _ in manager.process("Look it up.", user_id="u", session_id="s")]
        assert _marked(manager) == "google_search"
        assert await _stored_mark(manager) == "google_search"
    finally:
        await manager.cleanup()


async def test_a_native_adk_app_with_google_search_marks_its_conversation(tmp_path, monkeypatch):
    keys(monkeypatch, "google")
    route_gemini(monkeypatch, FakeAPI(gemini_api))
    path = tmp_path / "root_agent.yaml"
    path.write_text(textwrap.dedent(f"""
        agent_class: LlmAgent
        name: root
        model: {MODEL}
        instruction: You are root.
        tools:
          - name: google_search
        """), encoding="utf-8")
    manager = adk([], settings(default_model=MODEL), adk_config_path=str(path))
    try:
        [_ async for _ in manager.process("Look it up.", user_id="u", session_id="s")]
        assert _marked(manager) == "google_search"
        assert await _stored_mark(manager) == "google_search"
    finally:
        await manager.cleanup()


async def test_a_plain_agent_stays_unmarked(monkeypatch):
    keys(monkeypatch, "google")
    route_gemini(monkeypatch, FakeAPI(gemini_api))
    manager = adk(
        [AgentConfig(name="solo", prompt="Write.", include_state_tools=False)],
        settings(default_model=MODEL),
    )
    try:
        [_ async for _ in manager.process("Hello.", user_id="u", session_id="s")]
        assert _marked(manager) is None
        assert await _stored_mark(manager) == ""
    finally:
        await manager.cleanup()
