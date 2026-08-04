"""Entry point for the Research Demo application.

Usage:
    research-demo                    # console script (after pip install -e .)
    python -m research_demo
    python -m research_demo --session my-research   # resume a durable session
"""

import argparse
import asyncio

from .app import ResearchDemoApp


def main() -> None:
    """Run the Research Demo application."""
    parser = argparse.ArgumentParser(prog="research-demo")
    parser.add_argument(
        "--session",
        metavar="ID",
        default=None,
        help="Resume a durable session by id (sessions persist by default; "
        "list them with /sessions).",
    )
    args = parser.parse_args()
    asyncio.run(ResearchDemoApp(session_id=args.session).run())


if __name__ == "__main__":
    main()
