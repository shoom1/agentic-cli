"""Session handling for workflow management.

Encapsulates session lifecycle operations for ADK sessions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from google.adk.sessions import BaseSessionService, Session

from agentic_cli.logging import Loggers

if TYPE_CHECKING:
    pass

logger = Loggers.workflow()


class SessionHandler:
    """Handles session lifecycle operations.

    Encapsulates the logic for getting existing sessions or creating
    new ones, providing a clean interface for the WorkflowManager.

    Example:
        handler = SessionHandler(session_service, "my_app")
        session = await handler.get_or_create_session("user1", "session1")
    """

    def __init__(self, session_service: BaseSessionService, app_name: str):
        """Initialize session handler.

        Args:
            session_service: ADK session service for session management
            app_name: Application name for session scoping
        """
        self._session_service = session_service
        self._app_name = app_name

    @property
    def session_service(self) -> BaseSessionService:
        """Get the underlying session service."""
        return self._session_service

    async def get_or_create_session(
        self,
        user_id: str,
        session_id: str,
    ) -> Session:
        """Get existing session or create a new one.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            The session (existing or newly created)
        """
        session = await self._session_service.get_session(
            app_name=self._app_name,
            user_id=user_id,
            session_id=session_id,
        )

        if session is None:
            session = await self._session_service.create_session(
                app_name=self._app_name,
                user_id=user_id,
                session_id=session_id,
            )
            logger.debug("session_created", session_id=session_id)
        else:
            logger.debug("session_resumed", session_id=session_id)

        return session

    async def session_exists(self, user_id: str, session_id: str) -> bool:
        """Check if a session exists.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            True if the session exists
        """
        session = await self._session_service.get_session(
            app_name=self._app_name,
            user_id=user_id,
            session_id=session_id,
        )
        return session is not None
