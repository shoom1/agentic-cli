"""Entry point for the Research Demo application.

Usage:
    research-demo          # console script (after pip install -e .)
    python -m research_demo
"""

import asyncio

from .app import ResearchDemoApp


def main() -> None:
    """Run the Research Demo application."""
    asyncio.run(ResearchDemoApp().run())


if __name__ == "__main__":
    main()
