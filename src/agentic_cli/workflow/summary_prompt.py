"""The prompt a conversation is summarized with, on both backends.

Both backends summarize with LangChain's ``DEFAULT_SUMMARY_PROMPT``, the
prompt Deep Agents' summarization uses. The ADK backend runs without
LangChain, which only the ``deepagents`` and ``langgraph`` extras install, so
the prompt is copied here word for word and Deep Agents is given the same
copy; a test holds it to LangChain's, so a change there shows. A summary
reaches the model as user text that starts with ``SUMMARY_PREFIX``, as Deep
Agents writes its own.

``summary_prompt`` fills the messages in by splitting the prompt at its
``{messages}`` placeholder, not with ``str.format``, so braces in the
messages or in a focus are text. A focus (``/compact <focus>``) is a block
spliced in just before the ``<messages>`` marker, which LangChain documents
as a contract for this. ``with_focus`` is that splice alone, for LangChain's
own helper, which fills the prompt with ``str.format``.

``SUMMARY_PROMPT`` is copied from LangChain
(``langchain/agents/middleware/summarization.py``, langchain 1.4.6), which is
distributed under the MIT License:

    MIT License

    Copyright (c) LangChain, Inc.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

#: LangChain's ``DEFAULT_SUMMARY_PROMPT`` (langchain 1.4.6), word for word.
SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to continue working toward your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.

You should structure your summary using the following sections. Each section acts as a checklist - you must populate it with relevant information or explicitly state "None" if there is nothing to report for that section:

## SESSION INTENT

What is the user's primary goal or request? What overall task are you trying to accomplish? This should be concise but complete enough to understand the purpose of the entire session.

## SUMMARY

Extract and record all of the most important context from the conversation history. Include important choices, conclusions, or strategies determined during this conversation. Include the reasoning behind key decisions. Document any rejected options and why they were not pursued.

## ARTIFACTS

What artifacts, files, or resources were created, modified, or accessed during this conversation? For file modifications, list specific file paths and briefly describe the changes made to each. This section prevents silent loss of artifact information.

## NEXT STEPS

What specific tasks remain to be completed to achieve the session intent? What should you do next?

</instructions>

The user will message you with the full message history from which you'll extract context to create a replacement. Carefully read through it all and think deeply about what information is most important to your overall goal and should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501

#: The words a summary starts with, where the conversation it replaces began.
SUMMARY_PREFIX = "Here is a summary of the conversation to date:"

#: Where the messages go, and the marker a focus block is spliced in before.
_PLACEHOLDER = "{messages}"
_MARKER = "<messages>"


def summary_prompt(messages_text: str, focus: str | None = None) -> str:
    """The prompt that summarizes ``messages_text``, on ``focus`` when one is given.

    Args:
        messages_text: The conversation to summarize, as text.
        focus: What the summary should keep above all, in the user's words;
            None or blank for none.
    """
    head, _, tail = SUMMARY_PROMPT.partition(_PLACEHOLDER)
    if focus is not None and focus.strip():
        head = head.replace(_MARKER, _focus_block(focus) + _MARKER, 1)
    return f"{head}{messages_text}{tail}"


def with_focus(prompt: str, focus: str | None) -> str:
    """``prompt`` with a focus block just before its ``<messages>`` marker.

    For a prompt still to be filled with ``str.format``, as LangChain's
    summary helper fills it: braces in the focus are doubled, so they reach
    the model as written. Without a focus (None or blank), ``prompt`` as it is.
    """
    if focus is None or not focus.strip():
        return prompt
    block = _focus_block(focus).replace("{", "{{").replace("}", "}}")
    return prompt.replace(_MARKER, block + _MARKER, 1)


def _focus_block(focus: str) -> str:
    return f"<focus>\nFocus the summary on: {focus.strip()}\n</focus>\n\n"
