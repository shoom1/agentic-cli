"""A finished job waits for its conversation, wherever that conversation is kept.

A job that finished in a conversation, flagged ``resume_on_complete``, is told
in that conversation once it is opened (``cli/waiting_jobs.py``'s one rule);
it is told in another one, as unavailable, only when its conversation is
gone. A conversation belongs to one backend for its whole life, and the app
runs one backend at a time: after ``/resume`` onto another backend (or
``/new <backend>``, the picker, ``--continue``) the running manager's
``can_resume`` does not find a conversation another backend keeps. The rule
then asks the other installed backends' readers before calling it gone: a
conversation one of them keeps means the job waits (not told here, not in the
status bar's ``↻N``, not in the line an opened conversation says, not taken
by ``/jobs resume`` here). ``/resume`` into the job's conversation then tells
it there, on its own backend.

The fakes' class runs the app on fake managers over each backend's stored
conversations (``tests/cli/test_resume_on_its_backend.py``); the last runs the
real ADK and Deep Agents managers on sqlite.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("google.adk")
pytest.importorskip("deepagents")

from agentic_cli.cli.message_processor import TurnResult, TurnStatus  # noqa: E402
from agentic_cli.tools.jobs.manager import JobState  # noqa: E402
from agentic_cli.workflow.service_registry import JOB_MANAGER  # noqa: E402
from tests.cli import test_resume_on_its_backend as across  # noqa: E402
from tests.cli.test_compact_and_new import quiet_app, said  # noqa: E402,F401
from tests.cli.test_resume_on_its_backend import backends, on_adk, project  # noqa: E402,F401
from tests.cli.test_resume_session import replayed  # noqa: E402
from tests.cli.test_waiting_jobs import Jobs, finished  # noqa: E402
from tests.workflow.deepagents.fakes import ScriptedModels, ai  # noqa: E402
from tests.workflow.test_adk_compaction import Claude, says, serve  # noqa: E402


async def keeps(manager, record) -> bool:
    """A fake manager's ``can_resume``: whether it keeps the job's conversation."""
    return record.session_id in manager.stored


@pytest.fixture
def jobs_on_adk(backends):
    """An app in ``a1`` on ADK; ``d1`` is kept by Deep Agents. Its jobs:
    one of ``d1``'s, and one whose conversation no backend keeps."""
    backends.store("adk", "a1", "On ADK", 2)
    backends.store("deepagents", "d1", "On Deep Agents", 2, hours=1)
    return backends


async def started(backends, monkeypatch) -> tuple:
    """The app, its jobs (the run's: every manager it builds has them) and
    the jobs it told, in order."""
    jobs = Jobs([])
    build = backends.manager

    def manager(backend: str):
        built = build(backend)
        built.job_manager = jobs
        built.can_resume = lambda record: keeps(built, record)
        return built

    monkeypatch.setattr(backends, "manager", manager)
    app, _ = await on_adk(backends)
    me = app.settings.default_user
    jobs.records += [finished("theirs", "d1", me), finished("gone", "nowhere", me)]
    told: list[str] = []

    async def process_resume(*, record, claim, **_):
        if await claim():
            told.append(record.job_id)
        return TurnResult(TurnStatus.COMPLETED)

    app._message_processor.process_resume = process_resume
    return app, jobs, told


class TestOnTheFakes:
    async def test_jobs_resume_tells_the_gone_one_alone(
        self, jobs_on_adk, monkeypatch
    ):
        app, jobs, told = await started(jobs_on_adk, monkeypatch)

        await app.process_input("/jobs resume")

        assert told == ["gone"] and jobs.claimed == ["gone"]

    async def test_the_status_bar_counts_the_gone_one_alone(
        self, jobs_on_adk, monkeypatch
    ):
        app, _, _ = await started(jobs_on_adk, monkeypatch)
        monitor = app._new_job_monitor()

        assert await monitor.poll_once() == "↻1: /jobs resume"

    async def test_the_line_an_opened_conversation_says_counts_it_alone(
        self, jobs_on_adk, monkeypatch
    ):
        app, _, _ = await started(jobs_on_adk, monkeypatch)

        await app._name_its_waiting_jobs()

        assert said(app.session, "add_message") == [
            "A finished job waits for this conversation: /jobs resume tells it now."
        ]

    async def test_the_status_bar_keeps_the_readers_answer_a_while(
        self, jobs_on_adk, monkeypatch
    ):
        app, _, _ = await started(jobs_on_adk, monkeypatch)
        monitor = app._new_job_monitor()
        before = len(jobs_on_adk.readers)
        for _ in range(3):
            await monitor.poll_once()

        looks = jobs_on_adk.readers[before:]
        assert [reader.backend for reader in looks] == ["deepagents"] * 2  # each job once

    async def test_a_backend_that_cannot_be_read_counts_as_keeping_it(
        self, jobs_on_adk, monkeypatch
    ):
        """Unknown is no "gone": the job waits rather than be told here as
        unavailable."""
        jobs_on_adk.unreadable["deepagents"] = RuntimeError("database is locked")
        app, jobs, told = await started(jobs_on_adk, monkeypatch)

        await app.process_input("/jobs resume")

        assert told == []

    async def test_a_store_broken_for_good_is_warned_of_once_by_the_background_looks(
        self, jobs_on_adk, monkeypatch
    ):
        """The status bar's recheck and ``/jobs resume`` look at the other
        backends for each waiting job, again and again: a store that cannot
        be read is warned of once (above the prompt), then logged at debug.
        A look the user asked for (``/sessions``) still warns."""
        from agentic_cli.cli import job_monitor, stored_conversations

        warned: list[str] = []

        class Recorder:
            def __getattr__(self, level: str):
                def log(event: str, **fields) -> None:
                    if level == "warning":
                        warned.append(event)

                return log

        jobs_on_adk.unreadable["deepagents"] = RuntimeError("file is not a database")
        app, _, told = await started(jobs_on_adk, monkeypatch)  # its /resume a1 warned
        monkeypatch.setattr(stored_conversations, "logger", Recorder())
        monkeypatch.setattr(job_monitor, "_CONVERSATION_RECHECK_S", 0.0)
        monitor = app._new_job_monitor()

        for _ in range(5):
            await monitor.poll_once()
        await app.process_input("/jobs resume")

        assert told == [] and warned == ["session_reader_failed"]
        await app.process_input("/sessions")
        assert warned == ["session_reader_failed"] * 2

    async def test_resuming_its_conversation_tells_it_there(
        self, jobs_on_adk, monkeypatch
    ):
        app, jobs, told = await started(jobs_on_adk, monkeypatch)
        app._settings.job_auto_resume = True

        await app.process_input("/resume d1")

        assert (app.session_id, app.session_backend) == ("d1", "deepagents")
        assert told == ["theirs", "gone"]  # its own, and the one whose conversation is gone


class TestOnTheRealBackends:
    """A job of a Deep Agents conversation, while ADK runs: it waits, and
    ``/resume`` into its conversation tells it there, on Deep Agents."""

    @pytest.fixture
    def real(self, project, monkeypatch):
        serve(monkeypatch, Claude(solo=[says("A1."), says("A2.")]))
        models = ScriptedModels(solo=[ai("D1."), ai("The job is done.")])
        monkeypatch.setattr(across.deepagents_manager, "build_chat_model", models)
        monkeypatch.setattr(across.memory_services, "build_embedding_service",
                            lambda settings: across.MockEmbeddingService(embedding_dim=16))
        return models

    async def test_it_waits_on_the_other_backend_and_is_told_in_its_own(self, real):
        app = await across.TestOnTheRealBackends.started()
        app._settings.job_auto_resume = True
        controller = app._workflow_controller
        try:
            await app.process_input("One on ADK.")
            on_adk = app.session_id
            await app.process_input("/new deepagents")
            await app.process_input("One on Deep Agents.")
            on_deepagents = app.session_id
            jobs = controller.workflow.services[JOB_MANAGER]
            job = jobs.submit(
                tool="job_list", backend="inprocess", spec={"target": lambda: "done"},
                resume_on_complete=True, session_id=on_deepagents,
                user_id=app.settings.default_user, call_id="c1", call_name="job_list",
            )
            for _ in range(500):
                record = await asyncio.to_thread(jobs.get, job.job_id)
                if record.state is JobState.SUCCEEDED:
                    break
                await asyncio.sleep(0.01)
            assert record.awaiting_resume

            await app.process_input(f"/resume {on_adk}")
            assert app.session_backend == "adk"
            record = await asyncio.to_thread(jobs.get, job.job_id)
            assert record.awaiting_resume, f"told on ADK: {record.resume_state}"
            assert not [text for kind, text in replayed(app) if "unavailable" in text]

            await app.process_input(f"/resume {on_deepagents}")
            assert app.session_backend == "deepagents"
            record = await asyncio.to_thread(jobs.get, job.job_id)
            assert record.resume_state == "delivered"
            assert "The job is done." in [text for kind, text in replayed(app)]
            assert said(app.session, "add_error") == []
        finally:
            await controller.close()
