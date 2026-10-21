"""``field_to_ui_item`` turns a settings field into thinking-prompt 0.4's form field.

0.4 renamed 0.3's settings items (``CheckboxItem`` is ``CheckboxField``, and
so on) and froze them, so a field is built once, complete: the
``description`` and ``max_width`` a caller wants are arguments, never
assigned afterwards. The type rules are 0.3's, and so are the sizes 0.3's
items had (a dropdown's height and width, a text field's edit width).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field
from thinking_prompt import CheckboxField, DropdownField, SelectField, TextField

from agentic_cli.cli.settings_introspection import field_to_ui_item


class Backend(str, Enum):
    ADK = "adk"
    DEEPAGENTS = "deepagents"


class Settings(BaseModel):
    flag: bool = Field(default=False, title="A Flag", description="Turns it on")
    maybe_flag: bool | None = None
    effort: Literal["low", "medium", "high"] = "medium"
    backend: Backend = Backend.ADK
    few: str = Field(default="a", json_schema_extra={"options": ["a", "b"]})
    many: str = Field(default="c", json_schema_extra={"options": list("abcdef")})
    name: str = "x"
    count: int = 3
    ratio: float = 0.5


def _field(key: str, current, **kwargs):
    return field_to_ui_item(key, Settings.model_fields[key], current, **kwargs)


# --- the type rules ------------------------------------------------------------------


def test_a_bool_is_a_checkbox():
    field = _field("flag", True)

    assert field == CheckboxField(key="flag", label="A Flag", description="Turns it on", default=True)


def test_an_optional_bool_is_a_checkbox():
    assert _field("maybe_flag", None) == CheckboxField(key="maybe_flag", label="Maybe Flag", default=False)


def test_a_literal_is_a_select_of_its_values():
    field = _field("effort", "high")

    assert field == SelectField(
        key="effort", label="Effort", options=["low", "medium", "high"], default="high"
    )
    assert _field("effort", None).default == "low"


def test_an_enum_is_a_select_of_its_members_values():
    field = _field("backend", Backend.DEEPAGENTS)

    assert field == SelectField(
        key="backend", label="Backend", options=["adk", "deepagents"], default="deepagents"
    )


@pytest.mark.parametrize(
    ("count", "kind"), [(5, SelectField), (6, DropdownField)], ids=["five", "six"]
)
def test_dynamic_options_are_a_select_up_to_five_and_a_dropdown_beyond(count, kind):
    options = [f"m{i}" for i in range(count)]

    field = _field("name", "m1", dynamic_options=options)

    assert type(field) is kind
    assert (field.key, field.options, field.default) == ("name", options, "m1")
    assert _field("name", None, dynamic_options=options).default == ""


@pytest.mark.parametrize(
    ("key", "kind"), [("few", SelectField), ("many", DropdownField)], ids=["two", "six"]
)
def test_options_in_the_schema_are_a_select_or_a_dropdown(key, kind):
    field = _field(key, "b")

    assert type(field) is kind
    assert field.options == Settings.model_fields[key].json_schema_extra["options"]
    assert field.default == "b"


@pytest.mark.parametrize(("key", "current"), [("name", "y"), ("count", 7), ("ratio", 0.25)])
def test_text_and_numbers_are_text_fields_holding_a_string(key, current):
    field = _field(key, current)

    assert field == TextField(key=key, label=key.title(), default=str(current))


# --- 0.3's sizes ---------------------------------------------------------------------


def test_a_dropdown_keeps_0_3_s_menu_size():
    field = _field("many", "c")

    assert (field.height, field.width, field.max_width) == (4, 15, None)


def test_a_text_field_keeps_0_3_s_edit_width():
    field = _field("name", "x")

    assert (field.edit_width, field.password) == (15, False)


# --- description= and max_width= -----------------------------------------------------


def test_a_description_replaces_the_field_s_own():
    assert _field("flag", True, description="Something else").description == "Something else"


def test_no_description_keeps_the_field_s_own():
    assert _field("flag", True, description=None).description == "Turns it on"


def test_max_width_lets_a_dropdown_grow_to_it():
    field = _field("name", "m1", dynamic_options=[f"m{i}" for i in range(6)], max_width=48)

    assert isinstance(field, DropdownField)
    assert (field.width, field.max_width) == (None, 48)


def test_a_select_has_no_width_to_set():
    options = ["a", "b"]

    assert _field("name", "a", dynamic_options=options, max_width=48) == _field(
        "name", "a", dynamic_options=options
    )
