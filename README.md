# Agentic CLI

A framework for building domain-specific agentic CLI applications powered by LLM agents.

## Features

- **CLI Framework**: Rich terminal UI with thinking boxes and markdown support
- **Workflow Management**: Agent orchestration using Google ADK
- **Generic Tools**: Web search, Python execution, knowledge base
- **Session Persistence**: Save and restore conversation sessions
- **Configuration**: Type-safe settings with pydantic-settings

## Installation

```bash
pip install agentic-cli
```

## Usage

Create a domain-specific CLI application by extending the base classes:

```python
from agentic_cli import BaseCLIApp, WorkflowManager
from thinking_prompt import AppInfo

class MyApp(BaseCLIApp):
    def get_app_info(self) -> AppInfo:
        return AppInfo(name="MyApp", version="0.1.0")

    def get_settings(self):
        return MySettings()

    def create_workflow_manager(self):
        return MyWorkflowManager(settings=self.settings)
```

## License

MIT
