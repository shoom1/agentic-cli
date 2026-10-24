"""Entry point for the Research Demo application.

Usage:
    research-demo                    # console script (after pip install -e .)
    python -m research_demo
    research-demo --continue         # the newest saved conversation
    research-demo --resume <id>      # a saved conversation by id
    research-demo --resume           # pick one
"""

from .app import ResearchDemoApp


def main() -> None:
    """Run the Research Demo application (``BaseCLIApp.main()``: the session flags)."""
    ResearchDemoApp.main()


if __name__ == "__main__":
    main()
