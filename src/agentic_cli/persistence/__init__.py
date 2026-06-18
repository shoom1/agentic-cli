"""Persistence package.

Session state is now persisted natively by each orchestrator's store (ADK
``DatabaseSessionService`` / LangGraph checkpointer), keyed by session id — see
``BaseWorkflowManager`` session methods. The legacy JSON ``SessionPersistence``
snapshot layer was removed in favor of those durable, full-fidelity stores.
"""
