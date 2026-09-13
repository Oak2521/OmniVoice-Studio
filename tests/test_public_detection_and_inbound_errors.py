"""Offline response-boundary regressions; no AudioSeal, GPU or gRPC startup.

Compile the unchanged production function bodies with injected service edges.
This exercises error dictionaries and route serialization without importing
model-loading modules; full framework/auth/lifecycle integration belongs to CI.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


BACKEND = Path(__file__).resolve().parents[1] / "backend"
DIAGNOSTIC = "synthetic-private-path/model.bin synthetic-secret=do-not-return"


def production_function(path, name, namespace, *, owner=None):
    tree = ast.parse((BACKEND / path).read_text(encoding="utf-8"))
    nodes = tree.body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(BACKEND / path), "exec"), namespace)
    return namespace[name]


class PublicErrorBoundaryTests(unittest.TestCase):
    def test_watermark_service_failure_does_not_escape_through_upload_route(self):
        def fail_detector():
            raise RuntimeError(DIAGNOSTIC)

        logger = logging.getLogger("test.watermark.boundary")
        detect = production_function("services/watermark.py", "detect_watermark", {
            "_check_available": lambda: True,
            "_get_detector": fail_detector,
            "logger": logger,
        })
        # Route receives the real service's failure dictionary, not a mock
        # success or an exception caught by the route's outer handler.
        route = production_function("api/routers/watermark.py", "detect_audio_watermark", {
            "File": lambda *args, **kwargs: None,
            "_check_available": lambda: True,
            "os": os, "tempfile": tempfile,
            "torchaudio": SimpleNamespace(load=lambda path: (object(), 16000)),
            "detect_watermark": detect,
        })

        class Upload:
            filename = "synthetic.wav"

            async def read(self):
                return b"synthetic bytes; audio decoder is injected"

        with self.assertLogs(logger, level="WARNING") as records:
            response = asyncio.run(route(Upload()))
        self.assertNotIn(DIAGNOSTIC, json.dumps(response))
        self.assertTrue(response["error"])
        self.assertFalse(response["is_watermarked"])
        self.assertIn(DIAGNOSTIC, "\n".join(records.output))

    def test_inbound_snapshot_keeps_diagnostic_private_after_stop(self):
        snapshot = production_function("worker/inbound/service.py", "snapshot", {
            "bind_host": lambda: "127.0.0.1",
            "bind_port": lambda: 7444,
            "enabled": lambda: False,
            "is_exposed": lambda host: False,
        }, owner="InboundNode")
        node = SimpleNamespace(
            _log=SimpleNamespace(snapshot=lambda: {"sessions": [], "events": []}),
            running=False, port=0, startup_error=DIAGNOSTIC,
            _credentials=None, keys=SimpleNamespace(list_keys=lambda: []),
        )
        # A failed start followed by disable retains the internal diagnostic.
        # Both the status GET and the disable route return this snapshot.
        response = snapshot(node)
        self.assertNotIn(DIAGNOSTIC, json.dumps(response))
        self.assertTrue(response["startup_error"])
        self.assertEqual(node.startup_error, DIAGNOSTIC)

    def test_inbound_healthy_snapshot_preserves_none(self):
        snapshot = production_function("worker/inbound/service.py", "snapshot", {
            "bind_host": lambda: "127.0.0.1", "bind_port": lambda: 7444,
            "enabled": lambda: False, "is_exposed": lambda host: False,
        }, owner="InboundNode")
        node = SimpleNamespace(
            _log=SimpleNamespace(snapshot=lambda: {}), running=False, port=0,
            startup_error=None, _credentials=None,
            keys=SimpleNamespace(list_keys=lambda: []),
        )
        self.assertIsNone(snapshot(node)["startup_error"])


if __name__ == "__main__":
    unittest.main()
