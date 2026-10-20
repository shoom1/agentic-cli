"""Agent Skills as LangChain tools, for the Deep Agents backend.

The tools the ADK ``SkillToolset`` offers through ``make_skill_toolset``
(``toolset.py``): ``list_skills``, ``load_skill`` and ``load_skill_resource``,
with ADK's names, descriptions, argument schemas, results and error texts,
and the instruction ADK adds to the agent's system prompt for them. They run
on the skills ``SkillStore`` loaded (ADK ``Skill`` objects; ADK is a core
dependency), so a resource is read from the skill in memory and no path is
resolved at call time.

An ADK error result (``{"error", "error_code"}``) is sent as a failed call:
the same JSON, with the tool message's ``status="error"``, which is how this
backend marks a failure (ADK reads it from the ``error`` key).

What differs from ADK:

- A binary resource: ADK answers with a status and then adds the bytes to the
  next request as a user message. Here the result carries them, after the same
  status and the same lead-in text.
- ``load_skill``'s result (the instructions) stays in the conversation on both
  backends, but here summarization may condense it; the model can then load
  the skill again.

Not carried over: ``run_skill_script`` (no code executor is wired in, as on
the ADK manager's path), ADK's unload, search and lifecycle modes, and the
``adk_additional_tools`` and ``adk_inject_state`` frontmatter keys.

Each tool object is bound, in the framework registry, to the definition ADK's
skill tools are bound to (``permissions.py``): both backends gate a skill
tool by the same contract (EXEMPT), and an object the framework did not build
is denied, whatever its name.
"""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
from collections.abc import Sequence
from typing import Any

from google.adk.skills.prompt import format_skills_as_xml
from langchain_core.tools import BaseTool, StructuredTool, ToolException

from agentic_cli.tools.registry import bind_tool_identity, get_registry
from agentic_cli.workflow.sessions import get_active_turn

LIST_SKILLS = "list_skills"
LOAD_SKILL = "load_skill"
LOAD_SKILL_RESOURCE = "load_skill_resource"

#: The skill tools, in the order ADK's toolset offers them.
SKILL_TOOL_NAMES = (LIST_SKILLS, LOAD_SKILL, LOAD_SKILL_RESOURCE)

#: What the agent's system prompt gets: the text ADK's ``SkillToolset`` adds
#: for these three tools (no prefix, no script tool), word for word.
SKILLS_INSTRUCTION = (
    "You can use specialized 'skills' to help you with complex tasks. You MUST"
    " use the skill tools to interact with these skills.\n\n"
    "Skills are folders of instructions and resources that extend your"
    " capabilities for specialized tasks. Each skill folder contains:\n"
    "- **SKILL.md** (required): The main instruction file with skill metadata"
    " and detailed markdown instructions.\n"
    "- **references/** (Optional): Additional documentation or examples for"
    " skill usage.\n"
    "- **assets/** (Optional): Templates, scripts or other resources used by the"
    " skill.\n"
    "- **scripts/** (Optional): Scripts bundled with the skill. You cannot run"
    " them; use `load_skill_resource` to read one and follow it yourself.\n\n"
    "This is very important:\n\n"
    "1. If a skill seems relevant to the current user query, you MUST use the"
    ' `load_skill` tool with `skill_name="<SKILL_NAME>"` to read its full'
    " instructions before proceeding.\n"
    "2. Once you have read the instructions, follow them exactly as documented"
    " before replying to the user. For example, If the instruction lists"
    " multiple steps, please make sure you complete all of them in order.\n"
    "3. The `load_skill_resource` tool is for viewing files within a skill's"
    " directory (e.g., `references/*`, `assets/*`, `scripts/*`). It is ONLY for"
    " skill-bundled files — do NOT use it to access documents or files provided"
    " by the user at runtime. Do NOT use other tools to access skill files.\n"
    "4. If `load_skill_resource` returns any error, do not retry any path."
    " Report the error to the user and stop.\n"
    "5. Loading a skill only retrieves its instructions; it does NOT complete"
    " your turn. After a `load_skill` call returns, continue in the SAME turn:"
    " call whatever tools the skill's steps require (search, data retrieval,"
    " render), then write your reply. Never end your turn with an empty"
    " response right after loading a skill.\n"
    "\n\nNote: The following tools are NOT available: `run_skill_script`. Do"
    " NOT call them. After loading a skill (if available), apply its"
    " instructions in context and write your final reply as normal model text."
    " Never wrap the user-facing answer inside a tool call.\n"
)

#: Each tool's description and argument schema, as ADK declares them.
_DECLARATIONS: dict[str, tuple[str, dict[str, Any]]] = {
    LIST_SKILLS: (
        "Lists all available skills with their names and descriptions.",
        {"type": "object", "properties": {}},
    ),
    LOAD_SKILL: (
        "Loads the SKILL.md instructions for a given skill.",
        {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill to load.",
                },
            },
            "required": ["skill_name"],
        },
    ),
    LOAD_SKILL_RESOURCE: (
        (
            "Loads a resource file (from references/, assets/, or scripts/) from"
            " within a skill."
        ),
        {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The name of the skill.",
                },
                "file_path": {
                    "type": "string",
                    "description": (
                        "The relative path to the resource (e.g.,"
                        " 'references/my_doc.md', 'assets/template.txt',"
                        " or 'scripts/setup.sh')."
                    ),
                },
            },
            "required": ["skill_name", "file_path"],
        },
    ),
}

#: ADK's status for a binary resource.
BINARY_FILE_DETECTED = (
    "Binary file detected. The content has been injected into the"
    " conversation history for you to analyze."
)

_NO_TURN = object()


def make_skill_tools(skills: Sequence[Any]) -> tuple[list[BaseTool], str]:
    """One agent's skills as its three skill tools, and its skills instruction.

    Args:
        skills: Loaded ADK ``Skill`` objects (``SkillStore.resolve``).

    Returns:
        The tools, bound to the framework's skill tool definitions, and the
        instruction to append to the agent's system prompt. No skills: no
        tools and no instruction.

    Raises:
        ValueError: Two skills share a name (ADK's ``SkillToolset`` refuses
            them too).
    """
    if not skills:
        return [], ""
    answers = _SkillAnswers(skills)
    tools = [
        _tool(LIST_SKILLS, answers.list_skills),
        _tool(LOAD_SKILL, answers.load_skill),
        _tool(LOAD_SKILL_RESOURCE, answers.load_skill_resource),
    ]
    # The definitions are registered when the ``skills`` package is imported,
    # which importing this module guarantees.
    registry = get_registry()
    for tool in tools:
        definition = registry.get(tool.name)
        if definition is not None:
            bind_tool_identity(tool, definition)
    return tools, SKILLS_INSTRUCTION


def _tool(name: str, answer: Any) -> StructuredTool:
    description, schema = _DECLARATIONS[name]
    return StructuredTool(
        name=name,
        description=description,
        # ADK's schema as is; LangChain passes the call's arguments unvalidated,
        # as ADK does, so a missing one gets ADK's error result.
        args_schema=copy.deepcopy(schema),
        coroutine=answer,
        handle_tool_error=_error_content,
    )


class _Failed(ToolException):
    """A skill tool's error result, sent as a failed call."""

    def __init__(self, error: str, error_code: str) -> None:
        super().__init__(error)
        self.result = {"error": error, "error_code": error_code}


def _error_content(error: ToolException) -> str:
    result = getattr(error, "result", None) or {"error": str(error)}
    return json.dumps(result, ensure_ascii=False)


class _SkillAnswers:
    """What one agent's skill tools answer, from its skills in memory."""

    def __init__(self, skills: Sequence[Any]) -> None:
        self._skills: dict[str, Any] = {}
        for skill in skills:
            if skill.name in self._skills:
                raise ValueError(f"Duplicate skill name '{skill.name}'.")
            self._skills[skill.name] = skill
        self._misses_turn: object = _NO_TURN
        self._misses = 0

    # Unknown arguments are ignored, as ADK ignores them.

    async def list_skills(self, **_: Any) -> str:
        return format_skills_as_xml(list(self._skills.values()))

    async def load_skill(self, skill_name: Any = None, **_: Any) -> dict[str, Any]:
        if not skill_name:
            raise _Failed("Argument 'skill_name' is required.", "INVALID_ARGUMENTS")
        skill = self._skill(skill_name)
        return {
            "skill_name": skill_name,
            "instructions": skill.instructions,
            # The JSON ADK's function response is sent as.
            "frontmatter": skill.frontmatter.model_dump(mode="json"),
        }

    async def load_skill_resource(
        self, skill_name: Any = None, file_path: Any = None, **_: Any
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if not skill_name or not file_path:
            errors = []
            if not skill_name:
                errors.append("Argument 'skill_name' is required.")
            if not file_path:
                errors.append("Argument 'file_path' is required.")
            raise _Failed("\n".join(errors), "INVALID_ARGUMENTS")
        skill = self._skill(skill_name)
        content = _resource(skill, file_path)
        if content is None:
            misses = self._miss()
            if misses > 1:
                raise _Failed(
                    f"Resource '{file_path}' not found in skill '{skill_name}'."
                    f" This is resource lookup failure #{misses} this"
                    " invocation. Do not retry any path — report the error to"
                    " the user and stop.",
                    "RESOURCE_NOT_FOUND_FATAL",
                )
            raise _Failed(
                f"Resource '{file_path}' not found in skill '{skill_name}'.",
                "RESOURCE_NOT_FOUND",
            )
        if isinstance(content, bytes):
            return _binary(skill_name, file_path, content)
        return {"skill_name": skill_name, "file_path": file_path, "content": content}

    def _skill(self, skill_name: str) -> Any:
        skill = self._skills.get(skill_name)
        if skill is None:
            raise _Failed(f"Skill '{skill_name}' not found.", "SKILL_NOT_FOUND")
        return skill

    def _miss(self) -> int:
        """Count a resource lookup that found nothing; this turn's count so far.

        ADK counts them per invocation, which is a turn, and makes the second
        one fatal so the model stops guessing paths. A turn here is the active
        turn's ``SessionRef``, a new object every turn; two turns running at
        once on one manager restart each other's count. Each agent counts its
        own (ADK's count is the session's, shared by its agents): a sub-agent
        sees only its brief, not its coordinator's misses.
        """
        turn = get_active_turn()
        if turn is not self._misses_turn:
            self._misses_turn, self._misses = turn, 0
        self._misses += 1
        return self._misses


def _resource(skill: Any, file_path: str) -> str | bytes | None:
    """``file_path``'s content in ``skill``, as ADK looks it up."""
    if file_path.startswith("references/"):
        return skill.resources.get_reference(file_path[len("references/"):])
    if file_path.startswith("assets/"):
        return skill.resources.get_asset(file_path[len("assets/"):])
    if file_path.startswith("scripts/"):
        script = skill.resources.get_script(file_path[len("scripts/"):])
        return None if script is None else script.src
    raise _Failed(
        "Path must start with 'references/', 'assets/', or 'scripts/'.",
        "INVALID_RESOURCE_PATH",
    )


def _binary(skill_name: str, file_path: str, data: bytes) -> list[dict[str, Any]]:
    """ADK's status, then the bytes as ADK adds them to the request."""
    status = {
        "skill_name": skill_name,
        "file_path": file_path,
        "status": BINARY_FILE_DETECTED,
    }
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    return [
        {"type": "text", "text": json.dumps(status, ensure_ascii=False)},
        {"type": "text", "text": f"The content of binary file '{file_path}' is:"},
        {
            "type": "image" if mime_type.startswith("image/") else "file",
            "base64": base64.b64encode(data).decode("ascii"),
            "mime_type": mime_type,
        },
    ]
