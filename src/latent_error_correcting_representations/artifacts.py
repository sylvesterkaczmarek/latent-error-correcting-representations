"""Finite JSON evidence and exclusive, explicitly completed experiment outputs."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

MAX_JSON_BYTES = 32 * 1024 * 1024


def canonical(value):
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )
    if len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("JSON evidence exceeds 32 MiB")
    return text


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out

    def invalid(value):
        raise ValueError("nonfinite JSON constant")

    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON evidence exceeds 32 MiB")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid
        )
        canonical(value)
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("invalid or excessively nested JSON evidence") from exc
    return value


def write_new_json(path, value, *, compact=False):
    """Publish complete bytes without replacing an existing file or symlink."""
    if type(compact) is not bool:
        raise ValueError("compact must be a boolean")
    validated = canonical(value)  # Validate before touching the destination.
    text = (
        validated
        if compact
        else json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    )
    content = (text + "\n").encode("utf-8")
    if len(content) > MAX_JSON_BYTES:
        raise ValueError("formatted JSON evidence exceeds 32 MiB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix="." + path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        # The exclusive hard-link publication is atomic on supported local filesystems.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


@contextmanager
def experiment_output(path, plan):
    """A reused destination is an error. Interrupted directories lack completion.json."""
    path = Path(path)
    canonical(plan)
    path.mkdir(parents=True, exist_ok=False)
    write_new_json(path / "plan.json", plan)
    try:
        yield path
        files = {
            p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*"))
            if p.is_file()
        }
        write_new_json(path / "completion.json", {"status": "COMPLETE", "files": files})
    except BaseException as exc:
        # Failure evidence is retained, never presented as a completed experiment.
        try:
            write_new_json(
                path / "failure.json",
                {"status": "FAILED", "exception": type(exc).__name__},
            )
        except OSError:
            pass
        raise
