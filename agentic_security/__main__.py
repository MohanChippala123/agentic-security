"""Command-line entrypoint.

    python -m agentic_security serve --port 8000
"""

from __future__ import annotations

import argparse
import sys

# Windows consoles default to cp1252; force UTF-8 where supported.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    # Load .env from project root so GOOGLE_CLIENT_ID et al are available
    _env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    if _k and not os.environ.get(_k):
                        os.environ[_k] = _v
    except OSError:
        pass

    import uvicorn

    port = int(os.environ.get("PORT", args.port))
    host = os.environ.get("HOST", args.host)
    uvicorn.run("agentic_security.api.server:app",
                host=host, port=port, reload=args.reload,
                workers=args.workers)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentic_security",
        description="AI agent & API-key protection: prompt-injection firewall + key gateway",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Run the HTTP API + protection UI")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.add_argument("--workers", type=int, default=1)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
