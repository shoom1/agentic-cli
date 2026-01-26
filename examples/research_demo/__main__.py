"""Entry point for the Research Demo application.

Usage:
    conda run -n agenticcli python -m examples.research_demo
"""

import asyncio

from examples.research_demo.app import ResearchDemoApp


def main() -> None:
    """Run the Research Demo application."""
    asyncio.run(ResearchDemoApp().run())


if __name__ == "__main__":
    main()
