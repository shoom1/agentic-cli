"""Memory for agents: a knowledge base and a memory store.

One package with independent components. It imports nothing from the rest of
agentic-cli, and its components do not import each other
(``tests/memory/test_boundary.py`` checks both):

- ``agentic_cli.memory``: the memory store, and the embedding service the
  components share.
- ``agentic_cli.memory.kb``: the knowledge base.
"""
