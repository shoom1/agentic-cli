"""Entry point: python -m examples.hello_agent"""

import asyncio
from .app import HelloAgentApp


def main():
    app = HelloAgentApp()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
