"""``BaseCLIApp`` builds thinking-prompt 0.4's session, in agentic-cli's colors.

In 0.4 ``AppInfo`` is only the startup banner, and the session's behavior
(``theme``, ``echo_thinking``) is ``ThinkingPromptSession``'s. ``BaseCLIApp``
owns it, and an app recolors a theme through one hook, ``get_theme(name)``:
``"dark"`` is 0.4's dark theme with agentic-cli's colors, and any other name
is 0.4's own theme, since dark-tuned colors would be unreadable on a light
background. The session draws in the theme the user chose (``ui_theme``,
through ``get_theme``) and places dialogs where they chose
(``dialog_placement``). ``get_styles()`` is gone, and an app still defining it
fails at construction rather than starting with its colors silently dropped.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest
from prompt_toolkit.history import InMemoryHistory
from thinking_prompt import ThinkingPromptStyles

from agentic_cli.cli import app as app_module
from agentic_cli.cli.app import BaseCLIApp, SlashCommandCompleter
from agentic_cli.config import BaseSettings

# The colors agentic-cli has always drawn with: its dark theme.
AGENTIC_COLORS = {
    "thinking_box": "fg:#6c757d",
    "thinking_box_border": "fg:#495057",
    "thinking_box_hint": "fg:#6c757d italic",
    "user_message": "fg:#17a2b8",
    "assistant_message": "fg:#e9ecef",
    "system_message": "fg:#6c757d italic",
    "error_message": "fg:#dc3545 bold",
    "warning_message": "fg:#ffc107",
    "success_message": "fg:#28a745",
}


@pytest.fixture(autouse=True)
def hermetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A temporary cwd and HOME, and an app that leaves no global logging or
    warning hook behind."""
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for variable in ("AGENTIC_UI_THEME", "AGENTIC_DIALOG_PLACEMENT"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(app_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_install_adk_notice_hook", lambda: None)


def _app(cls: type[BaseCLIApp] = BaseCLIApp, **settings: str) -> BaseCLIApp:
    from agentic_cli.cli import AppInfo

    return cls(
        app_info=AppInfo(name="Theme test", version="0"),
        agent_configs=[],
        settings=BaseSettings(session_store="memory", **settings),
    )


def _bare(cls: type[BaseCLIApp] = BaseCLIApp) -> BaseCLIApp:
    """An app object for methods that read nothing it builds."""
    return cls.__new__(cls)


# --- get_theme ---------------------------------------------------------------------


def test_dark_is_thinking_prompt_s_dark_theme_in_agentic_cli_s_colors():
    theme = _bare().get_theme("dark")

    assert isinstance(theme, ThinkingPromptStyles)
    dark = ThinkingPromptStyles.dark()
    for field in dataclasses.fields(ThinkingPromptStyles):
        expected = AGENTIC_COLORS.get(field.name, getattr(dark, field.name))
        assert getattr(theme, field.name) == expected, field.name


def test_each_call_returns_a_theme_of_its_own():
    """A session may change the instance it holds; another must not see it."""
    app = _bare()

    assert app.get_theme("dark") is not app.get_theme("dark")


@pytest.mark.parametrize("name", ["light", "mono", "terminal", "auto"])
def test_every_other_theme_is_thinking_prompt_s_own(name):
    assert _bare().get_theme(name) == name


# --- the session -------------------------------------------------------------------


def test_the_session_gets_the_dark_theme_and_today_s_behavior():
    with patch.object(app_module, "ThinkingPromptSession") as session_cls:
        app = _app()

    (call,) = session_cls.call_args_list
    assert call.args == ()
    kwargs = dict(call.kwargs)
    theme = kwargs.pop("theme")
    history = kwargs.pop("history")
    completer = kwargs.pop("completer")
    assert theme == app.get_theme("dark")
    assert isinstance(history, InMemoryHistory)
    assert isinstance(completer, SlashCommandCompleter)
    assert "settings" in completer.commands
    assert kwargs == {
        "message": ">>> ",
        "app_info": app._app_info,
        "complete_while_typing": True,
        "completions_menu_height": 8,
        "enable_status_bar": True,
        "status_text": "Ctrl+C: cancel | Ctrl+D: exit | /help: commands",
        "echo_input": False,
        "echo_thinking": False,
        "dialog_placement": "box",
    }
    assert app.session is session_cls.return_value


@pytest.mark.parametrize("theme", ["light", "mono", "terminal", "auto"])
def test_the_session_takes_the_user_s_theme_and_placement(theme):
    with patch.object(app_module, "ThinkingPromptSession") as session_cls:
        _app(ui_theme=theme, dialog_placement="inline")

    kwargs = session_cls.call_args.kwargs
    assert (kwargs["theme"], kwargs["dialog_placement"]) == (theme, "inline")


def test_the_user_s_theme_goes_through_get_theme():
    """An app's colors for a theme apply to the one the user chose."""
    asked: list[str] = []

    class App(BaseCLIApp):
        def get_theme(self, name: str) -> str | ThinkingPromptStyles:
            asked.append(name)
            return super().get_theme(name)

    app = _app(App, ui_theme="mono")

    assert asked == ["mono"]
    assert app.session.styles == ThinkingPromptStyles.mono()


def test_the_real_session_places_dialogs_where_the_user_chose():
    assert _app().session.dialog_placement == "box"
    assert _app(dialog_placement="inline").session.dialog_placement == "inline"


def test_an_app_s_own_theme_reaches_the_session():
    mine = dataclasses.replace(ThinkingPromptStyles.dark(), thinking_box="fg:#9b59b6")

    class App(BaseCLIApp):
        def get_theme(self, name: str) -> str | ThinkingPromptStyles:
            return mine if name == "dark" else super().get_theme(name)

    assert _app(App).session.styles is mine


def test_the_real_session_draws_in_agentic_cli_s_colors():
    styles = _app().session.styles

    for name, value in AGENTIC_COLORS.items():
        assert getattr(styles, name) == value, name


# --- the removed get_styles ----------------------------------------------------------


def test_an_app_still_defining_get_styles_fails_and_is_told_of_get_theme():
    class Old(BaseCLIApp):
        def get_styles(self) -> ThinkingPromptStyles:
            return ThinkingPromptStyles(thinking_box="fg:#9b59b6")

    with patch.object(app_module, "ThinkingPromptSession") as session_cls:
        with pytest.raises(TypeError) as raised:
            _app(Old)

    assert str(raised.value) == (
        'get_styles() was replaced by get_theme(name) in agentic-cli 0.7.0: return '
        'your colors for "dark" from get_theme() (see the CHANGELOG)'
    )
    session_cls.assert_not_called()


def test_a_get_styles_inherited_from_an_app_s_own_base_fails_too():
    class Base(BaseCLIApp):
        def get_styles(self) -> ThinkingPromptStyles:
            return ThinkingPromptStyles()

    class App(Base):
        pass

    with pytest.raises(TypeError, match=r"get_theme\(name\)"):
        _app(App)


def test_the_old_styles_helpers_are_gone():
    assert not hasattr(BaseCLIApp, "get_styles")
    assert not hasattr(app_module, "get_default_styles")


# --- AppInfo -----------------------------------------------------------------------


def test_app_info_is_thinking_prompt_s_banner():
    import thinking_prompt

    from agentic_cli.cli import AppInfo

    assert AppInfo is thinking_prompt.AppInfo
    assert [f.name for f in dataclasses.fields(AppInfo)] == ["name", "version", "welcome_message"]
