"""`ml_imports` must not call a torchaudio API that may not be there (#1931).

`torchaudio.set_audio_backend()` was removed in torchaudio 2.9. `soundfile` had
been the only backend since 2.0, so the call was already a no-op — but
unguarded it raises `AttributeError` inside the `ml_imports` startup phase, and
a failure there takes the whole backend down: the desktop app sits on "starting
backend" forever and `/health` stays 503.

That is not hypothetical. RTX 50-series (Blackwell, sm_120) owners have no
choice but to move off the pinned torch 2.8.0, which carries no sm_120 kernels,
and the torch 2.9.x they land on brings torchaudio 2.9 with it. The one group
forced to upgrade met a hard startup crash for a line that does nothing.

The guard is one `hasattr`. This test is what keeps it: a cleanup pass that
sees a no-op call and "simplifies" it by deleting the condition would restore
the crash for every Blackwell user, and no test in the suite would notice —
CI runs the pinned torch, where the attribute still exists.
"""

import ast
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"

# APIs that a supported torchaudio release has already removed. Reading a
# missing attribute is an AttributeError, so each of these must be reached only
# behind a check that it exists.
REMOVED_TORCHAUDIO_APIS = {"set_audio_backend", "get_audio_backend", "list_audio_backends"}


def _guard_names(test: ast.AST) -> set:
    """Attribute names an `if` condition proves present.

    Recognises `hasattr(torchaudio, "x")` and `getattr(torchaudio, "x", None)`,
    including inside a boolean combination, which is how a real guard is
    written when it also checks something else.
    """
    names = set()
    for node in ast.walk(test):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in ("hasattr", "getattr") or len(node.args) < 2:
            continue
        target = node.args[1]
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            names.add(target.value)
    return names


def _unguarded_calls(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in REMOVED_TORCHAUDIO_APIS:
            continue
        if not (isinstance(node.value, ast.Name) and node.value.id == "torchaudio"):
            continue
        cursor, guarded = node, False
        while cursor in parents:
            cursor = parents[cursor]
            # Inside a try/except that would catch the AttributeError is also
            # a legitimate way to survive the removal.
            if isinstance(cursor, ast.Try):
                guarded = True
                break
            if isinstance(cursor, ast.If) and node.attr in _guard_names(cursor.test):
                guarded = True
                break
        if not guarded:
            bad.append((node.lineno, node.attr))
    return bad


def test_no_unguarded_removed_torchaudio_api():
    offenders = {}
    for path in sorted(BACKEND.rglob("*.py")):
        bad = _unguarded_calls(path)
        if bad:
            offenders[str(path.relative_to(BACKEND.parent))] = bad
    assert not offenders, (
        f"These reach a torchaudio API that torchaudio 2.9 removed, with "
        f"nothing proving it exists: {offenders}. Wrap it in "
        '`if hasattr(torchaudio, "..."):` — unguarded it is an AttributeError '
        "inside ml_imports, which takes the backend down on every RTX "
        "50-series machine (#1931)."
    )
