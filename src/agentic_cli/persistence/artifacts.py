"""Artifact persistence for agentic CLI applications.

Manages saving and loading of artifacts (plans, code, documentation) to disk.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_cli.config import BaseSettings


class ArtifactType(Enum):
    """Types of artifacts that can be persisted."""

    PLAN = "plan"
    CODE = "code"
    DOCUMENTATION = "documentation"
    TEST = "test"
    REPORT = "report"


@dataclass
class Artifact:
    """Represents a persistable artifact."""

    name: str
    artifact_type: ArtifactType
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert artifact to dictionary for serialization."""
        return {
            "name": self.name,
            "artifact_type": self.artifact_type.value,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Artifact":
        """Create artifact from dictionary."""
        return cls(
            name=data["name"],
            artifact_type=ArtifactType(data["artifact_type"]),
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


class ArtifactManager:
    """Manages artifact persistence to disk.

    Artifacts are stored in a workspace directory with the following structure:
        {workspace_dir}/workspace/
        ├── plans/
        │   └── {plan_name}.md
        ├── code/
        │   └── {code_name}.py
        ├── docs/
        │   └── {doc_name}.md
        ├── tests/
        │   └── {test_name}.py
        └── .artifacts.json  # Metadata index
    """

    SUBDIRS = {
        ArtifactType.PLAN: "plans",
        ArtifactType.CODE: "code",
        ArtifactType.DOCUMENTATION: "docs",
        ArtifactType.TEST: "tests",
        ArtifactType.REPORT: "reports",
    }

    EXTENSIONS = {
        ArtifactType.PLAN: ".md",
        ArtifactType.CODE: ".py",
        ArtifactType.DOCUMENTATION: ".md",
        ArtifactType.TEST: ".py",
        ArtifactType.REPORT: ".md",
    }

    def __init__(self, settings: "BaseSettings"):
        """Initialize artifact manager.

        Args:
            settings: Application settings with workspace_dir
        """
        self.workspace_path = settings.artifacts_dir
        self._ensure_workspace_exists()

    def _ensure_workspace_exists(self) -> None:
        """Create workspace directory structure if it doesn't exist."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        for subdir in self.SUBDIRS.values():
            (self.workspace_path / subdir).mkdir(exist_ok=True)

    def _get_artifact_path(self, artifact: Artifact) -> Path:
        """Get the file path for an artifact."""
        subdir = self.SUBDIRS[artifact.artifact_type]
        extension = self.EXTENSIONS[artifact.artifact_type]
        # Sanitize name for filesystem
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in artifact.name
        )
        return self.workspace_path / subdir / f"{safe_name}{extension}"

    def _get_index_path(self) -> Path:
        """Get path to artifact index file."""
        return self.workspace_path / ".artifacts.json"

    def _load_index(self) -> dict:
        """Load artifact index from disk."""
        index_path = self._get_index_path()
        if index_path.exists():
            with open(index_path, "r") as f:
                return json.load(f)
        return {"artifacts": []}

    def _save_index(self, index: dict) -> None:
        """Save artifact index to disk."""
        index_path = self._get_index_path()
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def save(self, artifact: Artifact) -> Path:
        """Save an artifact to disk.

        Args:
            artifact: The artifact to save

        Returns:
            Path to the saved artifact file
        """
        # Update timestamp
        artifact.updated_at = datetime.now()

        # Save content to file
        file_path = self._get_artifact_path(artifact)
        file_path.write_text(artifact.content)

        # Update index
        index = self._load_index()
        # Remove existing entry with same name and type
        index["artifacts"] = [
            a
            for a in index["artifacts"]
            if not (
                a["name"] == artifact.name
                and a["artifact_type"] == artifact.artifact_type.value
            )
        ]
        # Add new entry
        artifact_entry = artifact.to_dict()
        artifact_entry["file_path"] = str(file_path)
        index["artifacts"].append(artifact_entry)
        self._save_index(index)

        return file_path

    def load(self, name: str, artifact_type: ArtifactType) -> Artifact | None:
        """Load an artifact from disk.

        Args:
            name: Artifact name
            artifact_type: Type of artifact

        Returns:
            Artifact if found, None otherwise
        """
        index = self._load_index()
        for entry in index["artifacts"]:
            if entry["name"] == name and entry["artifact_type"] == artifact_type.value:
                # Load content from file
                file_path = Path(entry["file_path"])
                if file_path.exists():
                    content = file_path.read_text()
                    artifact = Artifact.from_dict(entry)
                    artifact.content = content
                    return artifact
        return None

    def list_artifacts(
        self, artifact_type: ArtifactType | None = None
    ) -> list[Artifact]:
        """List all artifacts, optionally filtered by type.

        Args:
            artifact_type: Filter by type (None for all)

        Returns:
            List of artifacts
        """
        index = self._load_index()
        artifacts = []
        for entry in index["artifacts"]:
            if (
                artifact_type is None
                or entry["artifact_type"] == artifact_type.value
            ):
                file_path = Path(entry["file_path"])
                if file_path.exists():
                    artifact = Artifact.from_dict(entry)
                    artifact.content = file_path.read_text()
                    artifacts.append(artifact)
        return artifacts

    def delete(self, name: str, artifact_type: ArtifactType) -> bool:
        """Delete an artifact.

        Args:
            name: Artifact name
            artifact_type: Type of artifact

        Returns:
            True if deleted, False if not found
        """
        index = self._load_index()
        for i, entry in enumerate(index["artifacts"]):
            if entry["name"] == name and entry["artifact_type"] == artifact_type.value:
                # Delete file
                file_path = Path(entry["file_path"])
                if file_path.exists():
                    file_path.unlink()
                # Remove from index
                index["artifacts"].pop(i)
                self._save_index(index)
                return True
        return False

    def get_workspace_path(self) -> Path:
        """Get the workspace path."""
        return self.workspace_path
