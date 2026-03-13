# Unreleased

## Added
- Persistent task progress thinking box that stays visible across turns while tasks are active (order=100, near prompt)
- Ephemeral LLM events thinking box for tool calls and thinking status (order=0, above tasks)
- Cold-start restoration of task box from cached content on new turn

## Changed
- Task progress display moved from composite status line to dedicated thinking box
- Events box uses per-box `ctx.finish()` instead of deprecated session-level `finish_thinking()`
- `workflow_controller.py` initialization wait uses per-box finish
- Bumped thinking-prompt dependency to >=0.3.0 for multi-box support

## Fixed
- Task progress box now updates when the last task completes (end-of-stream progress emission)
- Improved `save_tasks` tool description to instruct LLM to always mark tasks as completed

## Removed
- `task_progress_display` and `_last_task_display_content` fields from `_EventProcessingState`
