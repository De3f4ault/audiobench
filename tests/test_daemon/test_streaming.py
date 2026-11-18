import asyncio
import json
from unittest.mock import patch

import pytest

from audiobench.daemon.server import _HANDLERS, _handle_connection


@pytest.fixture
def mock_server_handlers():
    original_handlers = dict(_HANDLERS)

    def sync_handler(args):
        if args.get("fail"):
            raise RuntimeError("Handler failed")
        return {"result": args.get("val", 42)}

    async def async_generator_handler(args):
        yield {"status": "progress", "step": "1", "pct": 0.5}
        yield {"status": "progress", "step": "2", "pct": 1.0}
        yield {"status": "ok", "data": {"result": "done"}}

    _HANDLERS["test_sync"] = sync_handler
    _HANDLERS["test_async_gen"] = async_generator_handler

    yield

    _HANDLERS.clear()
    _HANDLERS.update(original_handlers)

import pytest_asyncio


@pytest_asyncio.fixture
async def unix_server(tmp_path, mock_server_handlers):
    socket_path = tmp_path / "test.sock"
    server = await asyncio.start_unix_server(
        _handle_connection, path=str(socket_path)
    )
    yield socket_path
    server.close()
    await server.wait_closed()

@pytest.mark.asyncio
async def test_single_response_handler_emits_ok_frame(unix_server):
    reader, writer = await asyncio.open_unix_connection(str(unix_server))

    req = {"cmd": "test_sync", "args": {"val": 100}, "request_id": "r1"}
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await reader.readline()
    resp = json.loads(line)

    assert resp["status"] == "ok"
    assert resp["data"]["result"] == 100
    assert resp["request_id"] == "r1"

    writer.close()
    await writer.wait_closed()

@pytest.mark.asyncio
async def test_progress_frames_received_in_order(unix_server):
    reader, writer = await asyncio.open_unix_connection(str(unix_server))

    req = {"cmd": "test_async_gen", "args": {}, "request_id": "r2"}
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    frames = []
    for _ in range(3):
        line = await reader.readline()
        if not line:
            break
        frames.append(json.loads(line))

    assert len(frames) == 3
    assert frames[0]["status"] == "progress"
    assert frames[0]["pct"] == 0.5
    assert frames[1]["status"] == "progress"
    assert frames[1]["pct"] == 1.0
    assert frames[2]["status"] == "ok"
    assert frames[2]["data"]["result"] == "done"
    assert all(f.get("request_id") == "r2" for f in frames)

    writer.close()
    await writer.wait_closed()

@pytest.mark.asyncio
async def test_error_frame_on_handler_exception(unix_server):
    reader, writer = await asyncio.open_unix_connection(str(unix_server))

    req = {"cmd": "test_sync", "args": {"fail": True}, "request_id": "r3"}
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await reader.readline()
    resp = json.loads(line)

    assert resp["status"] == "error"
    assert resp["error"]["code"] == "OPERATION_FAILED"
    assert "Handler failed" in resp["error"]["message"]

    writer.close()
    await writer.wait_closed()

@pytest.mark.asyncio
async def test_connection_timeout_yields_error(tmp_path):
    socket_path = tmp_path / "test_timeout.sock"

    async def mock_wait_for(coro, timeout=None):
        coro.close()
        raise TimeoutError("timeout")

    with patch("audiobench.daemon.server.asyncio.wait_for", new=mock_wait_for):
        server = await asyncio.start_unix_server(
            _handle_connection, path=str(socket_path)
        )
        try:
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            line = await reader.readline()
            resp = json.loads(line)

            assert resp["status"] == "error"
            assert resp["error"]["code"] == "TIMEOUT"

            writer.close()
            await writer.wait_closed()
        finally:
            server.close()
            await server.wait_closed()

@pytest.mark.asyncio
async def test_backward_compat_ping(unix_server):
    reader, writer = await asyncio.open_unix_connection(str(unix_server))

    req = {"cmd": "ping", "args": {}, "request_id": "r4"}
    writer.write((json.dumps(req) + "\n").encode())
    await writer.drain()

    line = await reader.readline()
    resp = json.loads(line)

    assert resp["status"] == "ok"
    assert "alive" in resp["data"]

    writer.close()
    await writer.wait_closed()
