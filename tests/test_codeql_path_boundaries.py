"""Three CodeQL path chains, using temporary files and real boundary code.

Only ffmpeg/ffprobe execution is injected. No media models, accounts or sockets
are needed. The tiny probe ASGI app preserves the production route decorator
and real native-access dependency; it is not the full application middleware.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from test_public_detection_and_inbound_errors import BACKEND, production_function


@pytest.fixture
def retime(tmp_path, monkeypatch):
    import core.config as config
    from services import video_retime

    root = tmp_path / "dub"
    root.mkdir()
    monkeypatch.setattr(config, "DUB_DIR", str(root))
    writes = []

    async def fake_ffmpeg(command, **kwargs):
        target = Path(command[-1])
        writes.append(target)
        target.write_bytes(b"synthetic output")
        return 0, b"", b""

    monkeypatch.setattr(video_retime, "run_ffmpeg", fake_ffmpeg)
    return video_retime, root, writes


def render(service, target):
    asyncio.run(service.render_retimed_video(
        job_id=None, ffmpeg="injected", video_path="synthetic-input",
        chunks=[(0.0, 1.0, 1.5)], out_path=str(target), batch_size=1,
    ))


@pytest.mark.parametrize("kind", ["traversal", "absolute", "sibling-prefix", "root"])
def test_retime_rejects_non_file_workspace_targets(retime, kind):
    service, root, writes = retime
    targets = {
        "traversal": root / ".." / "escape.mp4",
        "absolute": root.parent / "outside.mp4",
        "sibling-prefix": root.parent / "dub-other" / "outside.mp4",
        "root": root,
    }
    with pytest.raises(service.RetimeError) as caught:
        render(service, targets[kind])
    assert caught.value.stage == "plan"
    assert writes == []


def test_retime_does_not_reuse_or_delete_preexisting_slice_directory(retime):
    service, root, writes = retime
    output = root / "render.mp4"
    previous = Path(str(output) + ".slices")
    previous.mkdir()
    sentinel = previous / "unrelated.txt"
    sentinel.write_text("must survive", encoding="utf-8")
    render(service, output)
    assert sentinel.read_text(encoding="utf-8") == "must survive"
    assert output.read_bytes() == b"synthetic output"
    assert writes[0].parent != previous
    assert not writes[0].parent.exists()  # this invocation's temporary files cleaned


def test_retime_cleans_only_its_own_partial_files_on_failure(retime, monkeypatch):
    service, root, writes = retime
    output = root / "failed.mp4"

    async def fail_ffmpeg(command, **kwargs):
        target = Path(command[-1])
        writes.append(target)
        target.write_bytes(b"partial")
        return 1, b"", b"synthetic ffmpeg failure"

    monkeypatch.setattr(service, "run_ffmpeg", fail_ffmpeg)
    with pytest.raises(service.RetimeError):
        render(service, output)
    assert len(writes) == 1
    assert not writes[0].parent.exists()
    assert not output.exists()


def symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        if os.name != "nt":
            pytest.fail(f"Required POSIX symlink boundary could not run: {exc}")
        pytest.skip(f"Host cannot create synthetic symlinks: {exc}")


def test_retime_ignores_preexisting_slice_symlink(retime):
    service, root, writes = retime
    outside = root.parent / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("must survive", encoding="utf-8")
    output = root / "render.mp4"
    link = Path(str(output) + ".slices")
    symlink_or_skip(link, outside, directory=True)
    render(service, output)
    assert list(outside.iterdir()) == [sentinel]
    assert all(Path(os.path.realpath(p)).is_relative_to(root) for p in writes)


def test_retime_rejects_output_symlink_outside_root(retime):
    service, root, writes = retime
    outside = root.parent / "outside.mp4"
    outside.write_bytes(b"must survive")
    link = root / "render.mp4"
    symlink_or_skip(link, outside)
    with pytest.raises(service.RetimeError):
        render(service, link)
    assert outside.read_bytes() == b"must survive"
    assert writes == []


def test_prepare_retime_rejects_root_before_probe(retime, monkeypatch):
    service, root, writes = retime

    async def unexpected_probe(path):
        pytest.fail("Invalid work path reached the media probe")

    monkeypatch.setattr(service, "probe_frame_rates", unexpected_probe)
    with pytest.raises(service.RetimeError) as caught:
        asyncio.run(service.prepare_smart_fit_video(
            job_id=None, ffmpeg="injected", video_path="synthetic-input",
            plan=[], orig_dur=1, track_dur=1, work_path=str(root),
        ))
    assert caught.value.stage == "plan"


@pytest.fixture
def cover(tmp_path, monkeypatch):
    import core.config as config
    from services.longform_render import validate_cover_image

    monkeypatch.setattr(config, "OUTPUTS_DIR", str(tmp_path))
    directory = tmp_path / "audiobook_covers"
    directory.mkdir()
    # Evaluate the production allowlist, not a second regex invented by a test.
    tree = ast.parse((BACKEND / "api/routers/audiobook.py").read_text(encoding="utf-8"))
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "_COVER_NAME_RE" for t in n.targets))
    namespace = {"re": re, "os": os}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "cover-allowlist", "exec"), namespace)
    safe = production_function("api/routers/audiobook.py", "_safe_cover_path", namespace)
    return directory, safe, validate_cover_image


def test_cover_rebuilds_from_basename_and_requires_existing_allowed_file(cover):
    directory, safe, validate = cover
    valid = directory / "0123456789ab.png"
    valid.write_bytes(b"synthetic image placeholder")
    # Directory components are discarded: these can only select this upload.
    for value in [valid.name, str(valid), "../../" + valid.name]:
        resolved = safe(value)
        assert resolved == str(valid.resolve())
        assert validate(resolved)
    for value in [None, "../private.txt", "0123456789ab.png/extra", "0123456789ab.png.exe",
                  "ffffffffffff.png", "0123456789ab.png\n"]:
        assert safe(value) is None


def test_cover_rejects_matching_filename_symlink_outside_uploads(cover):
    directory, safe, validate = cover
    outside = directory.parent / "outside.png"
    outside.write_bytes(b"synthetic outside file")
    link = directory / "0123456789ab.png"
    symlink_or_skip(link, outside)
    assert safe(link.name) is None
    assert not validate(safe(link.name))


@pytest.mark.parametrize("host,allowed", [("127.0.0.1", True), ("::1", True),
    ("192.0.2.1", False), ("10.0.0.1", False), ("::ffff:192.0.2.1", False)])
def test_probe_route_uses_true_loopback_even_in_server_mode(tmp_path, monkeypatch, host, allowed):
    from fastapi import APIRouter, Depends, FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from api.dependencies import require_native_access

    monkeypatch.setenv("OMNIVOICE_SERVER_MODE", "1")
    monkeypatch.setenv("OMNIVOICE_API_KEY", "synthetic-key")
    monkeypatch.setenv("OMNIVOICE_TRUSTED_NETWORKS", "0.0.0.0/0,::/0")
    calls = []

    async def spawn(*args, **kwargs):
        calls.append(args)

        async def communicate():
            return b'{"format":{"format_name":"synthetic"}}', b""

        return SimpleNamespace(returncode=0, communicate=communicate)

    namespace = {"APIRouter": APIRouter, "Depends": Depends, "HTTPException": HTTPException,
        "BaseModel": BaseModel, "require_native_access": require_native_access,
        "os": os, "json": json, "asyncio": asyncio, "find_ffprobe": lambda: "injected",
        "spawn_subprocess": spawn, "router": APIRouter()}
    tree = ast.parse((BACKEND / "api/routers/tools.py").read_text(encoding="utf-8"))
    # Keep the exact @router.post(... dependencies=[...]) and request model.
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef))
             and n.name in {"ProbeReq", "probe"}]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "production-probe-route", "exec"), namespace)
    app = FastAPI()
    app.include_router(namespace["router"])
    target = tmp_path / "synthetic.wav"
    target.write_bytes(b"placeholder")
    with TestClient(app, client=(host, 12345)) as client:
        response = client.post("/tools/probe", json={"path": str(target)},
            headers={"Authorization": "Bearer synthetic-key", "X-Forwarded-For": "127.0.0.1"})
    assert response.status_code == (200 if allowed else 403)
    assert len(calls) == int(allowed)
    if allowed:
        assert calls[0][-1] == str(target.resolve())
