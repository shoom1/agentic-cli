"""Introspect Pydantic settings and convert to UI controls.

Provides utilities to automatically generate thinking_prompt form fields
from Pydantic field definitions, using type annotations and metadata.

Type → form field mapping:
    - bool → CheckboxField
    - str → TextField (text input)
    - int, float → TextField (the value comes back as a string)
    - Literal["a", "b", "c"] → SelectField (options from literal)
    - an Enum (e.g. OrchestratorType) → SelectField (its members' values)
    - Field with json_schema_extra={"options": [...]} → SelectField/DropdownField

thinking_prompt's fields are frozen: everything a field shows is passed when
it is built.
"""

from enum import Enum
from typing import Any, get_args, get_origin, Literal

from pydantic.fields import FieldInfo
from thinking_prompt import CheckboxField, DropdownField, SelectField, TextField


def field_to_ui_item(
    key: str,
    field: FieldInfo,
    current_value: Any,
    dynamic_options: list[str] | None = None,
    *,
    description: str | None = None,
    max_width: int | None = None,
) -> CheckboxField | SelectField | DropdownField | TextField:
    """Convert a Pydantic field to a thinking_prompt form field.

    Automatically selects the appropriate field based on the field's
    type annotation and metadata.

    Args:
        key: Field name (used as the form field's key)
        field: Pydantic FieldInfo containing type and metadata
        current_value: Current value of the field
        dynamic_options: Optional list of options to use instead of type-derived options
                        (useful for fields like 'model' where options come from runtime)
        description: Shown under the label instead of the field's own
            description; None keeps the field's own.
        max_width: The widest a dropdown's menu grows, sized to its longest
            option; None keeps the menu's fixed width. A select ignores it.

    Returns:
        A CheckboxField, TextField, SelectField or DropdownField
    """
    # Get metadata
    label = field.title or key.replace("_", " ").title()
    if description is None:
        description = field.description or ""
    extra = field.json_schema_extra or {}

    def one_of(options: list[str], default: str) -> SelectField | DropdownField:
        """A select for a few options, a dropdown for more than five."""
        if len(options) > 5:
            size = {} if max_width is None else {"width": None, "max_width": max_width}
            return DropdownField(
                key=key,
                label=label,
                description=description,
                options=options,
                default=default,
                **size,
            )
        return SelectField(
            key=key,
            label=label,
            description=description,
            options=options,
            default=default,
        )

    # Get the annotation type
    annotation = field.annotation

    # Handle dynamic options (e.g., for model selection)
    if dynamic_options is not None:
        return one_of(
            dynamic_options, str(current_value) if current_value is not None else ""
        )

    # Check for Literal type (enum-like options)
    origin = get_origin(annotation)
    if origin is Literal:
        options = list(get_args(annotation))
        # Convert options to strings for UI
        options = [str(opt) for opt in options]
        return SelectField(
            key=key,
            label=label,
            description=description,
            options=options,
            default=str(current_value) if current_value is not None else options[0],
        )

    # An Enum offers its members' values, as a Literal offers its own: the
    # value is what a setting accepts and what the dialog writes back.
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        options = [str(member.value) for member in annotation]
        current = current_value.value if isinstance(current_value, Enum) else current_value
        return SelectField(
            key=key,
            label=label,
            description=description,
            options=options,
            default=str(current) if current is not None else options[0],
        )

    # Check for explicit options in json_schema_extra
    if "options" in extra:
        return one_of(
            extra["options"], str(current_value) if current_value is not None else ""
        )

    # Check for bool
    if annotation is bool:
        return CheckboxField(
            key=key,
            label=label,
            description=description,
            default=bool(current_value) if current_value is not None else False,
        )

    # Check for Optional types (e.g., str | None)
    if origin is type(str | None):  # UnionType
        args = get_args(annotation)
        # Filter out NoneType
        non_none_types = [a for a in args if a is not type(None)]
        if len(non_none_types) == 1:
            # Recurse with the non-None type
            inner_type = non_none_types[0]
            if inner_type is bool:
                return CheckboxField(
                    key=key,
                    label=label,
                    description=description,
                    default=bool(current_value) if current_value is not None else False,
                )

    # Default to text input for str, int, float, and other types
    return TextField(
        key=key,
        label=label,
        description=description,
        default=str(current_value) if current_value is not None else "",
    )


def get_ui_order(field: FieldInfo) -> int:
    """Get sort order from field metadata.

    Fields with lower ui_order values appear first in the UI.
    Default order is 100 if not specified.

    Args:
        field: Pydantic FieldInfo

    Returns:
        Sort order integer
    """
    extra = field.json_schema_extra or {}
    return extra.get("ui_order", 100)


