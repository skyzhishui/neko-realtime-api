"""HTTP TTS pipeline tests - verifies P0 fix for transcript_done timing bug.

Background:
- Bug: ``await drain_task`` in `_stream_llm_events_to_tts_http` blocks the
  caller until ALL audio is drained. This delays `transcript.done` /
  `response.done` by tens of seconds for long replies, causing client
  idle-timeout disconnections (websockets 1011).
- Fix: drain runs as a background task; LLM-stream phase returns immediately
  after the LLM finishes producing text. Caller then sends `transcript.done`
  immediately, awaits drain via `_await_audio_drain()`, and only then sends
  `response.done`.

This suite mocks ``omni_client.stream_chat`` and the TTS pipeline so we can
simulate "LLM finishes fast, TTS is slow" and verify event ordering without
a live LLM/TTS backend.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace

import pytest

from server.session import RealtimeSession


# --------------------------------------------------------------------------- #
# Fakes / Helpers
# --------------------------------------------------------------------------- #


class _FakeProtocol:
    """Captures every send_* call with a wall-clock timestamp."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict, float]] = []
        self._t0 = time.monotonic()

    def _record(self, event_type: str, payload: dict | None = None) -> None:
        self.events.append((event_type, payload or {}, time.monotonic() - self._t0))

    async def send_transcript_delta(self, delta: str) -> None:
        self._record("transcript.delta", {"delta": delta})

    async def send_transcript_done(self, transcript: str = "") -> None:
        self._record("transcript.done", {"transcript": transcript})

    async def send_audio_delta(self, pcm: bytes) -> None:
        self._record("audio.delta", {"size": len(pcm)})

    async def send_response_done(
        self, resp_id: str, status: str = "completed", model: str | None = None,
    ) -> None:
        self._record(
            "response.done",
            {"resp_id": resp_id, "status": status, "model": model},
        )

    async def send_error(self, message: str) -> None:
        self._record("error", {"message": message})

    async def send_function_call_item_added(self, *a, **kw) -> None:  # pragma: no cover
        self._record("function_call.item_added", {})

    async def send_function_call_arguments_delta(self, *a, **kw) -> None:  # pragma: no cover
        self._record("function_call.arguments.delta", {})

    async def send_function_call_arguments_done(self, *a, **kw) -> None:  # pragma: no cover
        self._record("function_call.arguments.done", {})

    def reset_audio_residual(self) -> None:  # pragma: no cover
        """No-op: real ProtocolAdapter uses this to drop partial PCM bytes."""

    def types(self) -> list[str]:
        return [t for (t, _p, _ts) in self.events]


class _FakeOmniClient:
    """Yields a deterministic LLM-event stream (text deltas + finish)."""

    def __init__(self, deltas: list[str], delay_per_delta: float = 0.0) -> None:
        self._deltas = deltas
        self._delay = delay_per_delta

    async def stream_chat(self, **kwargs):
        for d in self._deltas:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield {"type": "text", "delta": d}
        yield {"type": "finish", "finish_reason": "stop"}


class _SlowTTSPipeline:
    """TTS that yields N PCM chunks with configurable per-chunk delay.

    Models the production-bug scenario: LLM finishes fast but TTS audio takes
    much longer to fully drain. We compress this with seconds -> milliseconds
    so tests run quickly, while preserving the relative ordering invariants.
    """

    def __init__(self, chunks_per_sentence: int = 5, chunk_delay_s: float = 0.05) -> None:
        self.voice = "test_voice"
        self._chunks = chunks_per_sentence
        self._delay = chunk_delay_s

    async def stream_tts(self, sentence: str, **kwargs):
        for _ in range(self._chunks):
            await asyncio.sleep(self._delay)
            yield b"\x00\x01\x02\x03" * 16  # 64-byte chunk


class _FakeConfig:
    def __init__(
        self,
        max_concurrent: int = 1,
        min_sub_sentence_len: int = 6,
        tts_model: str = "Qwen3-TTS",
    ) -> None:
        self._mc = max_concurrent
        self._msl = min_sub_sentence_len
        self._tts_model = tts_model

    def get(self, *path, default=None):
        if path == ("tts_pipeline", "min_sub_sentence_len"):
            return self._msl
        if path == ("tts_pipeline", "max_concurrent_tts"):
            return self._mc
        if path == ("services", "tts", "model"):
            return self._tts_model
        if path == ("services", "omni", "max_tokens"):
            return 4096
        if path == ("services", "omni", "timeout_s"):
            return 30
        return default


def _make_session(
    *,
    deltas: list[str],
    chunks_per_sentence: int = 5,
    chunk_delay_s: float = 0.05,
    delta_delay_s: float = 0.0,
    max_concurrent: int = 1,
):
    """Build a minimal RealtimeSession-like object exposing only what
    ``_stream_llm_events_to_tts_http`` touches.

    Bypasses ``__init__`` (which loads heavy models) and binds the real
    method on a SimpleNamespace via the descriptor protocol.
    """
    fake = SimpleNamespace()
    fake.protocol = _FakeProtocol()
    fake.omni_client = _FakeOmniClient(deltas=deltas, delay_per_delta=delta_delay_s)
    fake.config = _FakeConfig(max_concurrent=max_concurrent)
    fake.session_config = SimpleNamespace(
        voice="test_voice",
        temperature=0.7,
        repetition_penalty=1.0,
        tools=None,
        enable_search=False,
        _model="test-model",
    )
    fake.directive_stripper = None
    fake.gsv_tts_pipeline = None
    fake.tts_pipeline = _SlowTTSPipeline(
        chunks_per_sentence=chunks_per_sentence,
        chunk_delay_s=chunk_delay_s,
    )
    fake._audio_drain_task = None
    fake._audio_tts_tasks = None
    fake._pipeline_epoch = 0

    fake._llm_stream_kwargs = lambda: {
        "temperature": 0.7,
        "repetition_penalty": 1.0,
        "max_tokens": 4096,
        "timeout_s": 30,
        "tools": None,
        "enable_search": False,
    }

    # Bind real methods via descriptor protocol
    fake._stream_llm_events_to_tts_http = (
        RealtimeSession._stream_llm_events_to_tts_http.__get__(fake, type(fake))
    )
    if hasattr(RealtimeSession, "_await_audio_drain"):
        fake._await_audio_drain = (
            RealtimeSession._await_audio_drain.__get__(fake, type(fake))
        )
    if hasattr(RealtimeSession, "_cleanup_audio_tasks_on_error"):
        fake._cleanup_audio_tasks_on_error = (
            RealtimeSession._cleanup_audio_tasks_on_error.__get__(fake, type(fake))
        )
    return fake


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_transcript_done_sent_before_drain_completes():
    """C1: After fix, ``_stream_llm_events_to_tts_http`` must return BEFORE
    all audio chunks have been drained, so the caller can send
    ``transcript.done`` while drain is still in progress.

    Concretely the function must return as soon as the LLM stream ends
    (and all sentences are enqueued), without awaiting drain_task.  The
    drain_task must be exposed via ``self._audio_drain_task`` for the
    caller to await later.
    """
    # 3 short sentences, each TTS produces 5 chunks * 50ms = 250ms drain
    deltas = ["你好。", "今天天气不错。", "我们去散步吧。"]
    sess = _make_session(
        deltas=deltas,
        chunks_per_sentence=5,
        chunk_delay_s=0.05,
        delta_delay_s=0.01,
    )

    t0 = time.monotonic()
    full_text, _tc, fr = await sess._stream_llm_events_to_tts_http(
        messages=[{"role": "user", "content": "hi"}],
        resp_id="resp_test",
    )
    return_t = time.monotonic() - t0

    # Total drain work: 3 sentences * 5 chunks * 50ms = 750ms.
    # LLM stream itself: 3 deltas * 10ms = 30ms.
    # After fix: return <= ~200ms (LLM time + small overhead).
    # Before fix: return >= 750ms (must await drain).
    assert return_t < 0.5, (
        f"Function should return immediately after LLM stream ends, but "
        f"took {return_t:.3f}s -- likely still awaiting drain_task."
    )

    # drain_task must be exposed for caller to await
    assert sess._audio_drain_task is not None, (
        "After fix, `_stream_llm_events_to_tts_http` must expose drain_task "
        "via self._audio_drain_task so caller can await it after sending "
        "transcript.done."
    )
    assert isinstance(sess._audio_drain_task, asyncio.Task)
    assert sess._audio_tts_tasks is not None
    assert len(sess._audio_tts_tasks) == 3

    # Return values still correct
    assert full_text == "".join(deltas)
    assert fr == "stop"

    # At return time drain is still running (audio not all sent yet)
    audio_at_return = sum(
        1 for t, _p, _ts in sess.protocol.events if t == "audio.delta"
    )
    total_expected = 3 * 5
    assert audio_at_return < total_expected, (
        f"Drain should NOT be complete at return time, but {audio_at_return}/"
        f"{total_expected} chunks already sent."
    )

    # Now caller awaits drain -- all chunks should arrive
    await sess._audio_drain_task
    if sess._audio_tts_tasks:
        await asyncio.gather(*sess._audio_tts_tasks, return_exceptions=True)
    audio_total = sum(
        1 for t, _p, _ts in sess.protocol.events if t == "audio.delta"
    )
    assert audio_total == total_expected, (
        f"After awaiting drain, expected {total_expected} audio chunks but "
        f"got {audio_total}."
    )


@pytest.mark.asyncio
async def test_response_done_status_completed_via_await_audio_drain():
    """C4: ``_await_audio_drain()`` helper must await drain + tts_tasks and
    then clear the references so subsequent calls are idempotent."""
    deltas = ["短句。"]
    sess = _make_session(deltas=deltas, chunks_per_sentence=2, chunk_delay_s=0.02)

    await sess._stream_llm_events_to_tts_http(
        messages=[{"role": "user", "content": "hi"}],
        resp_id="resp_test",
    )
    assert sess._audio_drain_task is not None

    # Helper must exist
    assert hasattr(sess, "_await_audio_drain"), (
        "Session must expose `_await_audio_drain()` helper for caller to "
        "await drain + tts_tasks."
    )

    await sess._await_audio_drain()

    # References cleared
    assert sess._audio_drain_task is None
    assert sess._audio_tts_tasks is None

    # All audio drained
    audio_count = sum(
        1 for t, _p, _ts in sess.protocol.events if t == "audio.delta"
    )
    assert audio_count == 1 * 2

    # Idempotent
    await sess._await_audio_drain()


@pytest.mark.asyncio
async def test_cancel_during_drain_stops_audio():
    """C6: When the caller awaits ``_await_audio_drain()`` and gets
    cancelled mid-flight (e.g., user pressed cancel during long audio
    drain), the drain_task and tts_tasks must be cancelled, and no more
    audio is sent past a small in-flight tolerance.

    This emulates the real-world flow:
      1. _stream_llm_events_to_tts_http returns (LLM finished, drain ongoing)
      2. caller sends transcript.done
      3. caller awaits _await_audio_drain()  <-- gets cancelled here
      4. drain_task.cancel() must stop further audio.delta
    """
    deltas = ["第一句。", "第二句。", "第三句。"]
    sess = _make_session(
        deltas=deltas,
        chunks_per_sentence=10,
        chunk_delay_s=0.02,  # 200ms per sentence × 3 = 600ms drain
        delta_delay_s=0.005,
    )

    # Phase 1: run pipeline (LLM finishes fast, drain in progress)
    await sess._stream_llm_events_to_tts_http(
        messages=[{"role": "user", "content": "hi"}],
        resp_id="resp_test",
    )
    assert sess._audio_drain_task is not None

    # Phase 2: simulate caller awaiting drain, then getting cancelled
    drain_task = sess._audio_drain_task

    async def _await_then_cancellable():
        await sess._await_audio_drain()

    waiter = asyncio.create_task(_await_then_cancellable())
    await asyncio.sleep(0.05)  # let some chunks drain
    audio_at_cancel = sum(
        1 for t, _p, _ts in sess.protocol.events if t == "audio.delta"
    )

    # Cancel the waiter and the drain_task explicitly (mimicking
    # _cancel_active_pipeline)
    waiter.cancel()
    if not drain_task.done():
        drain_task.cancel()
    try:
        await waiter
    except asyncio.CancelledError:
        pass

    # Wait long enough that drain WOULD have finished if not cancelled
    await asyncio.sleep(0.3)
    audio_after_wait = sum(
        1 for t, _p, _ts in sess.protocol.events if t == "audio.delta"
    )
    leaked = audio_after_wait - audio_at_cancel
    # B5 (in-flight chunk leak) is left for P1; tolerate <=2 chunks.
    assert leaked <= 2, (
        f"After cancel, drain leaked {leaked} more chunks "
        f"(expected <=2 in-flight tolerance)."
    )


@pytest.mark.asyncio
async def test_queue_maxsize_backpressure():
    """C8 (B3 fix): per-sentence Queue must specify ``maxsize`` to prevent
    unbounded memory growth when TTS is faster than the WebSocket drain.
    """
    src = inspect.getsource(RealtimeSession._stream_llm_events_to_tts_http)
    assert "asyncio.Queue(maxsize=" in src, (
        "B3 fix not applied: per-sentence queue must use "
        "`asyncio.Queue(maxsize=N)` to prevent OOM on long replies."
    )
    bare_queue_count = src.count("asyncio.Queue()")
    assert bare_queue_count == 0, (
        f"Found {bare_queue_count} bare `asyncio.Queue()` calls -- all "
        f"per-sentence queues must specify maxsize for B3 backpressure."
    )


def test_audio_drain_task_field_initialized_in_init():
    """The session must initialize ``_audio_drain_task`` and
    ``_audio_tts_tasks`` fields so helpers can read them safely before any
    pipeline run."""
    src = inspect.getsource(RealtimeSession.__init__)
    assert "_audio_drain_task" in src, (
        "RealtimeSession.__init__ must initialize self._audio_drain_task = None"
    )
    assert "_audio_tts_tasks" in src, (
        "RealtimeSession.__init__ must initialize self._audio_tts_tasks = None"
    )


@pytest.mark.asyncio
async def test_tool_loop_drain_serialization():
    """Multi-round tool-call loop must await drain between rounds, so the
    next round does not overwrite an in-flight ``self._audio_drain_task``.

    Verification: inspect the source of ``_stream_llm_with_tool_loop`` to
    confirm it calls ``self._await_audio_drain()`` (or equivalent) inside
    the per-round loop.
    """
    src = inspect.getsource(RealtimeSession._stream_llm_with_tool_loop)
    assert "_await_audio_drain" in src, (
        "_stream_llm_with_tool_loop must call self._await_audio_drain() "
        "between rounds to avoid leaking drain_task across iterations."
    )


@pytest.mark.asyncio
async def test_caller_sends_transcript_done_before_response_done():
    """Verify all three caller paths (Mode B / Mode A / text_input) send
    ``send_transcript_done`` BEFORE ``send_response_done``, with an
    intervening ``_await_audio_drain()`` call.

    Source-level verification (event-order is already covered by the
    end-to-end fix design; this test guards against future refactors that
    might re-introduce the bug pattern).
    """
    src = inspect.getsource(RealtimeSession)
    # Each of the three callers must contain the pattern:
    #   send_transcript_done(...)
    #   ... _await_audio_drain ...
    #   send_response_done(...)
    # We use a simple structural check: every send_response_done(status="completed", ...)
    # call site must be preceded (in the same function) by send_transcript_done
    # and _await_audio_drain.
    for caller_name in ("_process_mode_b", "_process_mode_a", "_process_text_input"):
        method = getattr(RealtimeSession, caller_name)
        body = inspect.getsource(method)
        assert "send_transcript_done" in body, (
            f"{caller_name} must send_transcript_done"
        )
        assert "_await_audio_drain" in body, (
            f"{caller_name} must call _await_audio_drain() before "
            f"send_response_done(status=completed)"
        )
        # Ordering: transcript_done before _await_audio_drain before response_done
        idx_tdone = body.find("send_transcript_done")
        idx_drain = body.find("_await_audio_drain")
        idx_rdone = body.find('send_response_done')
        assert 0 <= idx_tdone < idx_drain < idx_rdone, (
            f"{caller_name} ordering must be: send_transcript_done -> "
            f"_await_audio_drain -> send_response_done; got positions "
            f"tdone={idx_tdone} drain={idx_drain} rdone={idx_rdone}"
        )


# --------------------------------------------------------------------------- #
# Phase 2 fix: epoch + synchronous reaping (cancel-path leak / cross-round mix)
# --------------------------------------------------------------------------- #


def test_pipeline_epoch_field_initialized_in_init():
    """The session must initialize ``_pipeline_epoch`` to 0 so cancel/cleanup
    paths can rely on the field always existing.

    Phase 2 fix (epoch-based cross-round isolation): every cancel increments
    the epoch; sentence/drain tasks self-check their captured epoch against
    ``self._pipeline_epoch`` and exit early when they don't match.
    """
    src = inspect.getsource(RealtimeSession.__init__)
    assert "_pipeline_epoch" in src, (
        "RealtimeSession.__init__ must initialize self._pipeline_epoch = 0 "
        "for cross-round task isolation."
    )


def test_cancel_active_pipeline_increments_epoch():
    """``_cancel_active_pipeline`` must increment ``self._pipeline_epoch`` so
    in-flight sentence/drain tasks from the cancelled round exit on their
    next epoch self-check, instead of waiting for HTTP timeouts."""
    src = inspect.getsource(RealtimeSession._cancel_active_pipeline)
    assert "_pipeline_epoch" in src, (
        "_cancel_active_pipeline must touch self._pipeline_epoch (increment "
        "on cancel so old-round tasks self-exit)."
    )
    # The function must increment (not just read) the epoch
    assert "_pipeline_epoch +=" in src or "_pipeline_epoch +" in src, (
        "_cancel_active_pipeline must INCREMENT self._pipeline_epoch."
    )


def test_cancel_active_pipeline_synchronous_reap():
    """``_cancel_active_pipeline`` must SYNCHRONOUSLY await cancelled tasks
    (with timeout) instead of fire-and-forget ``asyncio.create_task``.

    Bug regression: prior to this fix, ``_cancel_active_pipeline`` used
    ``asyncio.create_task(_cleanup_drain(drain_task))`` (fire-and-forget),
    which left drain/tts tasks pending after cancel. Symptoms:
      - "Task was destroyed but it is pending!"
      - TTS chunks logged 30-40s after session cleanup
    """
    import ast
    src = inspect.getsource(RealtimeSession._cancel_active_pipeline)
    # Strip leading whitespace from method (inspect returns indented text)
    src_dedent = inspect.cleandoc(src) if src.lstrip().startswith("async") else src
    # Parse to AST and walk only the executable code (skips docstrings/comments)
    try:
        # textwrap.dedent so ast.parse can handle it
        import textwrap
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        # Fall back to raw substring (worse but tolerable)
        tree = None

    # Helper: collect Call nodes whose function is `asyncio.create_task`
    create_task_args: list[str] = []
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                # Match `asyncio.create_task(...)` only
                if (
                    isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "asyncio"
                    and fn.attr == "create_task"
                    and node.args
                ):
                    arg = node.args[0]
                    # If the arg is a Call whose func name starts with `_cleanup_`,
                    # we have a fire-and-forget reaper (the regression pattern).
                    if isinstance(arg, ast.Call):
                        called = arg.func
                        called_name = ""
                        if isinstance(called, ast.Name):
                            called_name = called.id
                        elif isinstance(called, ast.Attribute):
                            called_name = called.attr
                        if called_name.startswith("_cleanup_"):
                            create_task_args.append(called_name)

    assert not create_task_args, (
        f"_cancel_active_pipeline must NOT use fire-and-forget "
        f"`asyncio.create_task(_cleanup_*)`; found: {create_task_args}. "
        f"Use synchronous `await asyncio.gather(...)` instead."
    )

    # Must reap synchronously: gather + return_exceptions, or wait_for, or wait
    has_sync_reap = (
        ("await asyncio.gather" in src and "return_exceptions=True" in src)
        or "await asyncio.wait_for" in src
        or "await asyncio.wait(" in src
    )
    assert has_sync_reap, (
        "_cancel_active_pipeline must synchronously reap cancelled tasks "
        "with `await asyncio.gather(...)` / `await asyncio.wait_for(...)`."
    )


def test_cleanup_synchronous_reap_drain_and_tts_tasks():
    """``cleanup()`` must synchronously reap ``_audio_drain_task`` and
    ``_audio_tts_tasks``, not just ``_active_pipeline_task``.

    Bug regression: prior to this fix, ``cleanup()`` only awaited
    ``_active_pipeline_task`` and left drain/tts tasks pending → they kept
    running for 30-40 seconds after the session was destroyed, producing
    ``TTS streaming error`` and ``Task was destroyed but it is pending``.
    """
    src = inspect.getsource(RealtimeSession.cleanup)
    # Must reference both task collections (otherwise it's not reaping them)
    assert "_audio_drain_task" in src, (
        "cleanup() must explicitly handle self._audio_drain_task — "
        "leaving it dangling caused 40s post-cleanup TTS leakage."
    )
    assert "_audio_tts_tasks" in src, (
        "cleanup() must explicitly handle self._audio_tts_tasks — "
        "leaving them pending caused `Task was destroyed but it is pending`."
    )


@pytest.mark.asyncio
async def test_drain_loop_self_exits_when_epoch_changes():
    """Cross-round isolation (Bug B fix): when a new round starts (epoch
    increments), the OLD drain_task must self-exit on its next loop
    iteration and stop calling ``protocol.send_audio_delta``, even if its
    sentence queues still have pending chunks.

    Without this self-check, old-round audio chunks bleed into the new
    round's WebSocket events, causing audio mixing of two responses.
    """
    deltas = ["第一句很长的内容用来产生足够的音频。", "第二句更长的内容继续产生音频。"]
    sess = _make_session(
        deltas=deltas,
        chunks_per_sentence=20,   # 20 chunks * 30ms = 600ms per sentence
        chunk_delay_s=0.03,
        delta_delay_s=0.005,
    )
    # Ensure epoch field exists on the fake (mirrors __init__)
    sess._pipeline_epoch = 0

    # Run pipeline; drain starts in background
    await sess._stream_llm_events_to_tts_http(
        messages=[{"role": "user", "content": "hi"}],
        resp_id="resp_test",
    )
    drain_task = sess._audio_drain_task
    assert drain_task is not None and not drain_task.done()

    # Let some chunks drain
    await asyncio.sleep(0.1)
    audio_at_epoch_change = sum(
        1 for t, _p, _ts in sess.protocol.events if t == "audio.delta"
    )

    # Bump epoch — drain_loop should self-exit on next iteration.
    sess._pipeline_epoch += 1

    # Wait long enough that drain WOULD have finished if it ran to completion
    await asyncio.sleep(0.8)

    audio_after_epoch_change = sum(
        1 for t, _p, _ts in sess.protocol.events if t == "audio.delta"
    )
    leaked = audio_after_epoch_change - audio_at_epoch_change

    # drain_loop self-check is between `q.get()` and `send_audio_delta`,
    # so ≤1 in-flight chunk per active sentence is the worst case.
    # We tolerate ≤3 (2 sentences × in-flight + scheduling jitter).
    assert leaked <= 3, (
        f"After epoch change, drain_loop leaked {leaked} more audio chunks "
        f"to client (expected ≤3 in-flight). drain_loop must self-check "
        f"`if epoch != self._pipeline_epoch: return` before send_audio_delta."
    )

    # drain_task should be done (self-exited) within reasonable time
    await asyncio.sleep(0.2)
    assert drain_task.done(), (
        "drain_task must self-exit when epoch changes; it is still running."
    )


@pytest.mark.asyncio
async def test_sentence_task_self_exits_when_epoch_changes():
    """Cross-round isolation (Bug B fix): when a new round starts, OLD-round
    sentence tasks must self-exit on their next chunk yield and stop pushing
    audio into queues.

    Without this self-check, an old-round TTS HTTP request that's slow to
    return its first chunk will keep streaming into a queue whose drain has
    already moved past — wasting bandwidth and risking memory pressure."""
    # Long sentences with slow chunks — they take time
    deltas = ["内容内容内容内容内容内容。"]
    sess = _make_session(
        deltas=deltas,
        chunks_per_sentence=30,   # 30 chunks * 50ms = 1.5s of TTS
        chunk_delay_s=0.05,
        delta_delay_s=0.0,
    )
    sess._pipeline_epoch = 0

    await sess._stream_llm_events_to_tts_http(
        messages=[{"role": "user", "content": "hi"}],
        resp_id="resp_test",
    )
    tts_tasks = sess._audio_tts_tasks
    assert tts_tasks is not None and len(tts_tasks) == 1
    sentence_task = tts_tasks[0]
    assert not sentence_task.done()

    # Let it run briefly
    await asyncio.sleep(0.1)

    # Bump epoch — sentence task should self-exit on next chunk
    sess._pipeline_epoch += 1

    # Wait — sentence task should exit within ≤1 chunk delay (50ms) + slack
    await asyncio.sleep(0.3)

    assert sentence_task.done(), (
        "sentence task must self-exit when epoch changes; it is still running. "
        "Sentence task must check `if epoch != self._pipeline_epoch: return` "
        "between TTS chunks."
    )


@pytest.mark.asyncio
async def test_cancel_path_no_pending_tasks_after_reap():
    """End-to-end regression test for the 'Task was destroyed but it is
    pending' bug.

    Sequence:
      1. Start a pipeline (LLM streams, drain runs)
      2. Cancel via ``_cancel_active_pipeline`` — it must SYNCHRONOUSLY
         reap drain + tts tasks
      3. Verify all tasks are .done() immediately after the call returns
         (no fire-and-forget orphans)
    """
    deltas = ["话语一。", "话语二。", "话语三。"]
    sess = _make_session(
        deltas=deltas,
        chunks_per_sentence=20,
        chunk_delay_s=0.03,
        delta_delay_s=0.005,
    )
    sess._pipeline_epoch = 0

    # Wire up `_active_pipeline_task` so _cancel_active_pipeline has something to cancel
    pipeline_done = asyncio.Event()
    async def _pipeline_runner():
        try:
            await sess._stream_llm_events_to_tts_http(
                messages=[{"role": "user", "content": "hi"}],
                resp_id="resp_test",
            )
        finally:
            pipeline_done.set()
    sess._active_pipeline_task = asyncio.create_task(_pipeline_runner())
    sess.interruption = SimpleNamespace(
        is_generating=True, set_generating=lambda v: setattr(sess.interruption, "is_generating", v),
    )

    # Bind real _cancel_active_pipeline
    sess._cancel_active_pipeline = (
        RealtimeSession._cancel_active_pipeline.__get__(sess, type(sess))
    )

    # Let drain start
    await asyncio.sleep(0.05)

    # Snapshot tasks before cancel
    drain_task = sess._audio_drain_task
    tts_tasks_snapshot = list(sess._audio_tts_tasks or [])

    # Synchronous reap
    await sess._cancel_active_pipeline("test_user_interrupt")

    # All cancelled tasks must be done IMMEDIATELY after _cancel_active_pipeline
    # returns. Fire-and-forget would leave them pending here.
    assert sess._active_pipeline_task is None
    if drain_task is not None:
        assert drain_task.done(), (
            "drain_task must be .done() after _cancel_active_pipeline returns "
            "(synchronous reap)."
        )
    for i, t in enumerate(tts_tasks_snapshot):
        assert t.done(), (
            f"sentence task #{i} must be .done() after _cancel_active_pipeline "
            f"returns (synchronous reap)."
        )

    # No "Task was destroyed but it is pending" possible — all tasks reaped.
