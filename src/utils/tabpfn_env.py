"""Set TABPFN_TOKEN before importing tabpfn (CLI, notebooks, Hydra child processes inherit env)."""

from __future__ import annotations

import os
from pathlib import Path


def ensure_tabpfn_token() -> None:
    """Set ``os.environ['TABPFN_TOKEN']`` for non-interactive TabPFN auth.

    **Files override shell env** (except ``TABPFN_TOKEN_FILE``): if you once exported a stale
    ``TABPFN_TOKEN`` in ``~/.bashrc``, TabPFN would keep using it and ignore ``~/.config/.../token``
    because ``browser_auth.get_cached_token`` checks the environment first. We therefore load
    disk / repo secrets *before* falling back to ``TABPFN_TOKEN`` / ``PRIORLABS_API_KEY`` in the
    environment.

    Order:
      1) ``TABPFN_TOKEN_FILE`` if set
      2) ``~/.config/tabpfn/token`` (single line, no newline)
      3) ``~/.tabpfn_token``
      4) Repo-root ``.env.local`` then ``.env`` (only ``TABPFN_TOKEN`` / ``PRIORLABS_API_KEY``)
      5) ``PRIORLABS_API_KEY`` then ``TABPFN_TOKEN`` from the environment

    Use the API key from https://ux.priorlabs.ai/account and accept the license on the Licenses tab.
    If verification still fails, rotate the key on the account page (old JWTs can be revoked).
    """
    repo_root = Path(__file__).resolve().parents[2]

    def _read_token_file(path: Path) -> str | None:
        try:
            if not path.is_file():
                return None
            tok = path.read_text(encoding="utf-8").strip().splitlines()
            if not tok:
                return None
            return tok[0].strip()
        except OSError:
            return None

    def _parse_env_file(path: Path) -> str | None:
        if not path.is_file():
            return None
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            if key not in ("TABPFN_TOKEN", "PRIORLABS_API_KEY"):
                continue
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            if val:
                return val
        return None

    chosen: str | None = None

    tf = (os.environ.get("TABPFN_TOKEN_FILE") or "").strip()
    if tf:
        chosen = _read_token_file(Path(tf).expanduser())
    if not chosen:
        chosen = _read_token_file(Path.home() / ".config" / "tabpfn" / "token")
    if not chosen:
        chosen = _read_token_file(Path.home() / ".tabpfn_token")
    if not chosen:
        for name in (".env.local", ".env"):
            chosen = _parse_env_file(repo_root / name)
            if chosen:
                break
    if not chosen:
        chosen = (os.environ.get("PRIORLABS_API_KEY") or "").strip() or None
    if not chosen:
        chosen = (os.environ.get("TABPFN_TOKEN") or "").strip() or None

    if chosen:
        os.environ["TABPFN_TOKEN"] = chosen
