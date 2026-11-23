"""Tool calls made from inside an execute_code cell fire pre_tool_call hooks under the calling
conversation's identity (#145870): without it a hook sees ``session_id=""`` and, in a
multiplexed gateway, can only fall back to the process-wide HERMES_SESSION_ID of whichever
conversation bound a turn last."""

import json
from unittest.mock import patch

import pytest

from model_tools import handle_function_call
from tools.code_execution_rpc import _default_dispatch
from tools.code_kernel import shutdown_all_kernels


@pytest.fixture(autouse=True)
def _local_kernel(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    shutdown_all_kernels()
    with patch("tools.code_execution_tool._load_config", return_value={"mode": "strict", "timeout": 30}):
        yield
    shutdown_all_kernels()


@pytest.fixture
def hook_calls():
    calls = []

    def _record(tool_name, args, **hook_kwargs):
        calls.append((tool_name, hook_kwargs))
        return None, None

    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=_record):
        yield calls


def _nested(calls, tool):
    return [kw for name, kw in calls if name == tool]


def test_nested_call_in_a_local_cell_carries_the_outer_session_and_turn(tmp_path, hook_calls):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    code = f"from hermes_tools import read_file\nprint(read_file({str(target)!r}))"

    result = json.loads(handle_function_call(
        "execute_code", {"code": code}, task_id="task-A", session_id="sess-A", turn_id="turn-A",
        conversation_id="conv-A", enabled_tools=["execute_code", "read_file"]))

    assert result["status"] == "success", result
    nested = _nested(hook_calls, "read_file")
    assert len(nested) == 1
    assert nested[0]["session_id"] == "sess-A"
    assert nested[0]["turn_id"] == "turn-A"
    assert nested[0]["conversation_id"] == "conv-A"


def test_nested_call_over_the_remote_rpc_dispatch_carries_the_outer_session_and_turn(tmp_path, hook_calls):
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")

    dispatch = _default_dispatch("task-A", "sess-A", "turn-A", "conv-A")
    assert "hello" in json.loads(dispatch("read_file", {"path": str(target)}))["content"]

    nested = _nested(hook_calls, "read_file")
    assert len(nested) == 1
    assert nested[0]["session_id"] == "sess-A"
    assert nested[0]["turn_id"] == "turn-A"
    assert nested[0]["conversation_id"] == "conv-A"


@pytest.mark.parametrize("route", ["kernel", "per_call"])
def test_both_remote_routes_hand_the_outer_identity_to_the_rpc_dispatch(route, monkeypatch):
    import inspect
    from contextlib import ExitStack

    from tests.tools.test_code_kernel_remote import ScriptedEnv, _cell, _spawn_ok_handlers
    from tools import code_execution_rpc
    from tools.code_kernel_remote import shutdown_all_remote_kernels

    real, seen = code_execution_rpc._default_dispatch, []

    def _recording(*args, **kwargs):
        seen.append(inspect.signature(real).bind(*args, **kwargs).arguments)
        return real(*args, **kwargs)

    handlers = [("command -v python3", lambda c: {"output": "OK\n", "returncode": 0})]
    if route == "kernel":
        handlers += _spawn_ok_handlers([_cell(stdout="ran\n")])
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    with ExitStack() as stack:
        stack.callback(shutdown_all_remote_kernels)
        stack.enter_context(patch("tools.code_execution_tool._get_or_create_env",
                                  return_value=(ScriptedEnv(handlers), "ssh")))
        stack.enter_context(patch("tools.code_execution_tool._remote_write"))
        stack.enter_context(patch("tools.code_kernel_remote._remote_write"))
        stack.enter_context(patch.object(code_execution_rpc, "_default_dispatch", side_effect=_recording))
        if route == "per_call":
            stack.enter_context(patch("tools.code_kernel_remote.execute_in_remote_kernel", return_value=None))
        result = json.loads(handle_function_call(
            "execute_code", {"code": "print(1)"}, task_id="task-A", session_id="sess-A", turn_id="turn-A",
            conversation_id="conv-A", enabled_tools=["execute_code", "read_file"]))

    assert result["status"] == "success", result
    assert [(a["session_id"], a["turn_id"], a["conversation_id"]) for a in seen] == [("sess-A", "turn-A", "conv-A")]


# ---------------------------------------------------------------------------
# Inner-call identity: an inner call is dispatched with ``<owner>#<n>`` and is the
# owner of its own post_tool_call. Without both halves the observer that opens a
# span on pre_tool_call can never close it, so every nested call ends up marked
# "Turn ended before this call's response/result was recorded."
# ---------------------------------------------------------------------------


def test_a_nested_call_carries_its_own_id_and_emits_both_hooks(tmp_path):
    """The whole contract through the real registry and a real kernel: an inner call inside a
    suppressing executor (how the agent loop runs every tool) still emits pre AND post, and both
    carry ``<owner>#1``. Fails if ``tool_call_id`` is dropped by ``registry.dispatch``'s
    signature filter, and fails if the leaf does not re-enable the post hook."""
    from hermes_cli import lifecycle, plugins
    import model_tools

    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")
    code = f"from hermes_tools import read_file\nprint(read_file({str(target)!r}))"
    events = []

    def _pre(tool_name, args, **hook_kwargs):
        events.append(("pre", tool_name, hook_kwargs.get("tool_call_id", "")))
        return None, None

    def _post(hook_name, **hook_kwargs):
        if hook_name == "post_tool_call":
            events.append(("post", hook_kwargs.get("tool_name"), hook_kwargs.get("tool_call_id", "")))

    with patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", side_effect=_pre), \
         patch.object(lifecycle, "has_hook", lambda name: name == "post_tool_call"), \
         patch.object(lifecycle, "invoke_hook", _post), \
         patch("tools.approval.check_execute_code_guard",
               return_value={"approved": True, "user_approved": False, "message": ""}):
        with model_tools.suppress_post_tool_call_hook():
            result = json.loads(handle_function_call(
                "execute_code", {"code": code}, task_id="task-A", tool_call_id="B1",
                session_id="sess-A", turn_id="turn-A", conversation_id="conv-A",
                enabled_tools=["execute_code", "read_file"]))

    assert result["status"] == "success", result
    assert [e for e in events if e[1] == "read_file"] == [
        ("pre", "read_file", "B1#1"),
        ("post", "read_file", "B1#1"),
    ]


def test_cell_authority_mints_one_id_per_call_and_clears_the_post_suppression():
    import model_tools
    from tools.code_kernel import CellAuthority

    seen = []

    def _fake(name, args, **kwargs):
        seen.append((name, kwargs.get("tool_call_id"), model_tools._post_tool_call_hook_suppressed.get()))
        return json.dumps({"ok": True})

    with patch.object(model_tools, "handle_function_call", _fake):
        with model_tools.suppress_post_tool_call_hook():
            authority = CellAuthority("task-A", "sess-A", "turn-A", "conv-A", "B1")
            authority.dispatch("web_search", {"query": "a"})
            authority.dispatch("web_search", {"query": "b"})

    assert seen == [("web_search", "B1#1", False), ("web_search", "B1#2", False)]


def test_cell_authority_without_an_owner_passes_no_id():
    import model_tools
    from tools.code_kernel import CellAuthority

    seen = []

    def _fake(name, args, **kwargs):
        seen.append(kwargs.get("tool_call_id"))
        return json.dumps({"ok": True})

    with patch.object(model_tools, "handle_function_call", _fake):
        CellAuthority("task-A").dispatch("web_search", {"query": "a"})

    assert seen == [None]


def test_the_rpc_dispatch_mints_only_when_an_owner_is_known():
    import model_tools

    seen = []

    def _fake(name, args, **kwargs):
        seen.append(kwargs.get("tool_call_id"))
        return json.dumps({"ok": True})

    with patch.object(model_tools, "handle_function_call", _fake):
        _default_dispatch("task-A")("web_search", {"query": "x"})
        owned = _default_dispatch("task-A", owner_tool_call_id="B2")
        owned("web_search", {"query": "y"})
        owned("web_search", {"query": "z"})

    assert seen == [None, "B2#1", "B2#2"]


def test_the_poll_loop_forwards_the_owner_and_a_refused_call_takes_no_id(tmp_path):
    """The remote/Docker transport end to end: the real file client allocates seq, the poll loop
    dispatches with ``<owner>#<n>``, and an over-budget refusal neither dispatches nor consumes a
    number."""
    import os
    import threading

    import model_tools
    from tests.tools.test_code_execution_rpc import _FakeFileRpcEnvironment
    from tools.code_execution_tool import _rpc_poll_loop, generate_hermes_tools_module

    rpc_dir = tmp_path / "rpc"
    rpc_dir.mkdir()
    token = "rpc-token-probe"
    env = _FakeFileRpcEnvironment(str(rpc_dir))
    tool_log, counter, stop = [], [0], threading.Event()
    seen = []

    def _fake(name, args, **kwargs):
        seen.append((name, kwargs.get("tool_call_id")))
        return json.dumps({"ok": 1})

    def _poll():
        with patch.object(model_tools, "handle_function_call", _fake):
            _rpc_poll_loop(
                env, str(rpc_dir), "task-A", tool_log, counter,
                max_tool_calls=2, allowed_tools=frozenset({"web_search"}),
                stop_event=stop, rpc_token=token, owner_tool_call_id="B7",
            )

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()
    namespace = {"__name__": "hermes_tools"}
    try:
        with patch.dict(os.environ, {"HERMES_RPC_DIR": str(rpc_dir), "HERMES_RPC_TOKEN": token}):
            exec(generate_hermes_tools_module(["web_search"], transport="file"), namespace)
            call = namespace["_call"]
            responses = [call("web_search", {"query": q}) for q in ("a", "b", "c")]
    finally:
        stop.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert seen == [("web_search", "B7#1"), ("web_search", "B7#2")]
    assert responses[0] == {"ok": 1} and responses[1] == {"ok": 1}
    assert "Tool call limit reached (2)" in json.dumps(responses[2])
    assert counter[0] == 2, "a refused call must not consume budget or an id"


IDENTITY_KEYS = {
    "task_id", "session_id", "tool_call_id", "turn_id", "api_request_id", "conversation_id",
    "parent_tool_call_id",
}

BASE_IDENTITY = {
    "task_id": "task-A", "session_id": "", "tool_call_id": "",
    "turn_id": "", "api_request_id": "", "conversation_id": "",
}


def test_the_identity_record_keeps_its_hook_contract():
    """T11 — the record gains exactly one field. RED on the base, which has six keys."""
    import model_tools

    kwargs = model_tools._CallIds("task-A").hook_kwargs()
    assert set(kwargs) == IDENTITY_KEYS, (
        "the identity record must carry exactly seven keys; "
        f"missing={sorted(IDENTITY_KEYS - set(kwargs))} extra={sorted(set(kwargs) - IDENTITY_KEYS)}"
    )
    assert kwargs["tool_call_id"] == ""
    assert kwargs["parent_tool_call_id"] == ""


def test_top_level_payload_carries_the_seventh_identity_key(tmp_path, hook_calls):
    """T3 — top-level, unidentified fixture: seven identity keys, both ids empty. RED on the base."""
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")

    result = json.loads(handle_function_call("read_file", {"path": str(target)}, task_id="task-A"))
    assert result.get("status", "success") == "success", result

    payloads = _nested(hook_calls, "read_file")
    assert len(payloads) == 1, payloads
    payload = payloads[0]

    identity = {k: v for k, v in payload.items() if k in IDENTITY_KEYS}
    assert set(identity) == IDENTITY_KEYS, (
        "the pre_tool_call payload must carry seven identity keys; "
        f"missing={sorted(IDENTITY_KEYS - set(identity))} extra={sorted(set(identity) - IDENTITY_KEYS)}"
    )
    assert identity["tool_call_id"] == ""
    assert identity["parent_tool_call_id"] == ""


@pytest.fixture
def post_payloads():
    """Capture what reaches the post_tool_call hook, which rebuilds its payload by hand."""
    seen = []

    def _invoke(hook_name, *args, **kwargs):
        seen.append((hook_name, kwargs))

    with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
         patch("hermes_cli.lifecycle.invoke_hook", side_effect=_invoke):
        yield seen


def test_ownerless_payload_delta_is_exactly_the_new_key(tmp_path, post_payloads):
    """T4b — the emitted post payload is today's dict plus one key, and nothing else differs. RED on the base."""
    target = tmp_path / "note.txt"
    target.write_text("hello\n", encoding="utf-8")

    result = json.loads(handle_function_call("read_file", {"path": str(target)}, task_id="task-A"))
    assert result.get("status", "success") == "success", result

    emitted = [kw for name, kw in post_payloads if name == "post_tool_call"]
    assert len(emitted) == 1, emitted
    payload = emitted[0]

    assert "parent_tool_call_id" in payload, (
        "the seventh identity key is absent from the emitted post_tool_call payload"
    )
    identity = {k: v for k, v in payload.items() if k in IDENTITY_KEYS}
    assert identity == {**BASE_IDENTITY, "parent_tool_call_id": ""}, identity


# ── W6 remainder: parentage through both dispatch paths, guards, consumers ────


def test_local_cell_child_carries_its_owner_as_parent(hook_calls):
    """T1 — CellAuthority child: tool_call_id == "<owner>#1", parent_tool_call_id == "<owner>"."""
    from tools.code_kernel import CellAuthority

    CellAuthority("task-A", "sess-A", "turn-A", "conv-A", "B1").dispatch("web_search", {"query": "a"})

    payloads = _nested(hook_calls, "web_search")
    assert len(payloads) == 1, payloads
    assert payloads[0]["tool_call_id"] == "B1#1"
    assert payloads[0]["parent_tool_call_id"] == "B1"


def test_remote_dispatch_child_carries_its_owner_as_parent(hook_calls):
    """T2 — the same two equalities through _default_dispatch."""
    _default_dispatch("task-A", "sess-A", "turn-A", "conv-A", "B7")("web_search", {"query": "b"})

    payloads = _nested(hook_calls, "web_search")
    assert len(payloads) == 1, payloads
    assert payloads[0]["tool_call_id"] == "B7#1"
    assert payloads[0]["parent_tool_call_id"] == "B7"


def test_ownerless_dispatch_passes_neither_id():
    """T4a — baseline guard: an ownerless call keeps today's call shape (no id keywords at all)."""
    import model_tools

    seen = []

    def _spy(*args, **kwargs):
        seen.append(kwargs)
        return '{"status": "success"}'

    with patch.object(model_tools, "handle_function_call", _spy):
        _default_dispatch("task-A")("web_search", {"query": "x"})

    assert len(seen) == 1, seen
    assert "tool_call_id" not in seen[0], seen[0]
    assert "parent_tool_call_id" not in seen[0], seen[0]


def test_interleaved_cells_keep_their_own_owners(hook_calls):
    """T5 — two owners interleaved: every child points at its own owner, never the sibling's."""
    from tools.code_kernel import CellAuthority

    a = CellAuthority("task-A", "sess-A", "turn-A", "conv-A", "B1")
    b = CellAuthority("task-A", "sess-A", "turn-A", "conv-A", "B2")
    a.dispatch("web_search", {"query": "a1"})
    b.dispatch("web_search", {"query": "b1"})
    a.dispatch("web_search", {"query": "a2"})
    b.dispatch("web_search", {"query": "b2"})

    pairs = [(p["tool_call_id"], p["parent_tool_call_id"]) for p in _nested(hook_calls, "web_search")]
    assert pairs == [("B1#1", "B1"), ("B2#1", "B2"), ("B1#2", "B1"), ("B2#2", "B2")], pairs


def test_nested_execution_is_refused_by_the_cell_sandbox(hook_calls):
    """T6 — conditional, and the branch that applies here is the refusal.

    §7 makes T6 red when nested execution is supported and a baseline guard when the runtime refuses it.
    The cell sandbox (SANDBOX_ALLOWED_TOOLS) does not include `execute_code`, so the refusal is the
    expected behaviour and asserting it is the correct outcome — recorded, not skipped.
    """
    code = "from hermes_tools import execute_code\nprint(execute_code('print(1)'))"
    result = json.loads(handle_function_call(
        "execute_code", {"code": code}, task_id="task-A", session_id="sess-A", turn_id="turn-A",
        conversation_id="conv-A", tool_call_id="B9", enabled_tools=["execute_code", "read_file"]))

    body = json.dumps(result).lower()
    # Measured branch (2026-11-23): hermes_tools does not expose execute_code, so the cell raises
    # ImportError and no nested dispatch happens. Asserted as the expected behaviour, not skipped.
    assert result.get("status") == "error", result
    assert "importerror" in body and "execute_code" in body, result
    assert result.get("tool_calls_made", 0) == 0, result
    # The only lifecycle event is the OUTER call; no nested entry is dispatched.
    outer = _nested(hook_calls, "execute_code")
    assert len(outer) == 1, outer
    assert outer[0]["tool_call_id"] == "B9"
    # NOTE: the outer event's parent value is deliberately NOT asserted here. §7 makes T6 a
    # base-independent guard when nesting is refused; asserting the seventh key would make it a
    # contract test and it would be red on the base for the wrong reason. That property is covered
    # by T3 (unidentified top-level) and by T8 (identified child) instead.


def test_strict_and_kwargs_consumers_receive_the_parent():
    """T8 — delivery at the callback level: a legacy explicit signature and a **kwargs consumer,
    on both pre_tool_call and post_tool_call, for one top-level call and one owned child.
    This is the test that fails if E4b (pre) or E3b (post) is missing."""
    from hermes_cli.plugins import get_plugin_manager

    mgr = get_plugin_manager()
    pre_legacy, pre_kwargs, post_legacy, post_kwargs = [], [], [], []

    def legacy_pre(tool_name, args, task_id="", session_id="", tool_call_id="", turn_id="",
                   api_request_id="", middleware_trace=None, conversation_id=""):
        pre_legacy.append({"tool_name": tool_name, "tool_call_id": tool_call_id})

    def legacy_post(tool_name, args, result, task_id="", session_id="", tool_call_id="", turn_id="",
                    api_request_id="", duration_ms=0, status=None, error_type=None, error_message=None,
                    middleware_trace=None, conversation_id=""):
        post_legacy.append({"tool_name": tool_name, "tool_call_id": tool_call_id})

    def kwargs_pre(**kw):
        pre_kwargs.append(kw)

    def kwargs_post(**kw):
        post_kwargs.append(kw)

    for name, cb in (("pre_tool_call", legacy_pre), ("pre_tool_call", kwargs_pre),
                     ("post_tool_call", legacy_post), ("post_tool_call", kwargs_post)):
        mgr._hooks.setdefault(name, []).append(cb)
    try:
        import os
        import tempfile
        tmp = tempfile.mkdtemp()
        # Two distinct files: a second read of the same path is deduplicated ("unchanged"), which
        # would make this test assert the wrong thing.
        top_path = os.path.join(tmp, "top.txt")
        child_path = os.path.join(tmp, "child.txt")
        for p in (top_path, child_path):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("hello\n")

        top = json.loads(handle_function_call("read_file", {"path": top_path}, task_id="task-A"))
        assert top.get("status", "success") == "success", top

        child = json.loads(handle_function_call(
            "read_file", {"path": child_path}, task_id="task-A", session_id="sess-A", turn_id="turn-A",
            conversation_id="conv-A", tool_call_id="B5#1", parent_tool_call_id="B5"))
        assert child.get("status", "success") == "success", child
    finally:
        for name, cb in (("pre_tool_call", legacy_pre), ("pre_tool_call", kwargs_pre),
                         ("post_tool_call", legacy_post), ("post_tool_call", kwargs_post)):
            mgr._hooks.get(name, []).remove(cb)

    assert len(pre_legacy) == 2 and len(post_legacy) == 2, (pre_legacy, post_legacy)
    assert len(pre_kwargs) == 2 and len(post_kwargs) == 2, (pre_kwargs, post_kwargs)

    assert pre_kwargs[0]["parent_tool_call_id"] == ""      # top-level, unidentified
    assert pre_kwargs[1]["parent_tool_call_id"] == "B5"    # owned child
    assert post_kwargs[0]["parent_tool_call_id"] == ""
    assert post_kwargs[1]["parent_tool_call_id"] == "B5"
    assert pre_kwargs[1]["tool_call_id"] == "B5#1"
    assert post_kwargs[1]["tool_call_id"] == "B5#1"
