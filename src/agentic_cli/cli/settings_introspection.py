"""Introspect Pydantic settings and convert to UI controls.

Provides utilities to automatically generate thinking_prompt UI items
from Pydantic field definitions, using type annotations and metadata.

Type → UI Control Mapping:
    - bool → CheckboxItem
    - str → TextItem (text input)
    - int, float → TextItem (with validation)
    - Literal["a", "b", "c"] → InlineSelectItem (options from literal)
    - Field with json_schema_extra={"options": [...]} → InlineSelectItem/DropdownItem
"""

from typing import Any, get_args, get_origin, Literal, TYPE_CHECKING

from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from thinking_prompt import CheckboxItem, InlineSelectItem, TextItem, DropdownItem


def field_to_ui_item(
    key: str,
    field: FieldInfo,
    current_value: Any,
    dynamic_options: list[str] | None = None,
) -> Any:
    """Convert a Pydantic field to a thinking_prompt UI item.

    Automatically selects the appropriate UI control based on the field's
    type annotation and metadata.

    Args:
        key: Field name (used as UI item key)
        field: Pydantic FieldInfo containing type and metadata
        current_value: Current value of the field
        dynamic_options: Optional list of options to use instead of type-derived options
                        (useful for fields like 'model' where options come from runtime)

    Returns:
        UI item (CheckboxItem, TextItem, InlineSelectItem, or DropdownItem)
    """
    # Import here to avoid circular imports and allow optional dependency
    from thinking_prompt import CheckboxItem, InlineSelectItem, TextItem, DropdownItem

    # Get metadata
    label = field.title or key.replace("_", " ").title()
    description = field.description or ""
    extra = field.json_schema_extra or {}

    # Get the annotation type
    annotation = field.annotation

    # Handle dynamic options (e.g., for model selection)
    if dynamic_options is not None:
        if len(dynamic_options) > 5:
            return DropdownItem(
                key=key,
                label=label,
                description=description,
                options=dynamic_options,
                default=str(current_value) if current_value is not None else "",
            )
        return InlineSelectItem(
            key=key,
            label=label,
            description=description,
            options=dynamic_options,
            default=str(current_value) if current_value is not None else "",
        )

    # Check for Literal type (enum-like options)
    origin = get_origin(annotation)
    if origin is Literal:
        options = list(get_args(annotation))
        # Convert options to strings for UI
        options = [str(opt) for opt in options]
        return InlineSelectItem(
            key=key,
            label=label,
            description=description,
            options=options,
            default=str(current_value) if current_value is not None else options[0],
        )

    # Check for explicit options in json_schema_extra
    if "options" in extra:
        options = extra["options"]
        if len(options) > 5:
            return DropdownItem(
                key=key,
                label=label,
                description=description,
                options=options,
                default=str(current_value) if current_value is not None else "",
            )
        return InlineSelectItem(
            key=key,
            label=label,
            description=description,
            options=options,
            default=str(current_value) if current_value is not None else "",
        )

    # Check for bool
    if annotation is bool:
        return CheckboxItem(
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
                return CheckboxItem(
                    key=key,
                    label=label,
                    description=description,
                    default=bool(current_value) if current_value is not None else False,
                )

    # Default to text input for str, int, float, and other types
    return TextItem(
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


def get_ui_section(field: FieldInfo) -> str | None:
    """Get UI section from field metadata.

    Allows grouping related settings in the UI.

    Args:
        field: Pydantic FieldInfo

    Returns:
        Section name or None if not specified
    """
    extra = field.json_schema_extra or {}
    return extra.get("ui_section")


def is_ui_hidden(field: FieldInfo) -> bool:
    """Check if field should be hidden from UI.

    Args:
        field: Pydantic FieldInfo

    Returns:
        True if field should be hidden
    """
    extra = field.json_schema_extra or {}
    return extra.get("ui_hidden", False)
