# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CLI entry point for coder_mcp.

Usage::

    python -m coder_mcp --port 8083 --backend apiary --apiary-url http://127.0.0.1:8080

Environment variables:
    APIARY_URL           Apiary daemon URL (default http://127.0.0.1:8080)
    APIARY_API_TOKEN     Bearer token for the Apiary daemon
    APIARY_WORKING_DIR   Default working directory inside sandbox (default /workspace)
    MCP_AUTH_TOKEN       If set, clients must present this Bearer token on SSE
"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from coder_mcp.server import ServerConfig, create_server

LOGGER = logging.getLogger(__name__)


def _install_uvloop() -> bool:
    try:
        import uvloop

        uvloop.install()
        LOGGER.info("Using uvloop event loop")
        return True
    except ImportError:
        LOGGER.info("uvloop not available, using default asyncio event loop")
        return False


def _build_backend_kwargs(args: argparse.Namespace) -> dict:
    """Translate CLI flags into backend-specific constructor kwargs.

    Currently only the ``apiary`` backend has CLI flags; future backends
    can add their own flags and mapping here.
    """
    if args.backend == "apiary":
        return {
            "base_url": args.apiary_url,
            "token": args.apiary_token,
            "working_dir": args.working_dir,
            "idle_timeout": args.idle_timeout,
        }
    return {}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _install_uvloop()

    parser = argparse.ArgumentParser(
        description="MCP server for the Coder agent with pluggable sandbox backends"
    )
    parser.add_argument("--host", default="0.0.0.0", help="SSE bind host")
    parser.add_argument("--port", type=int, default=8083, help="SSE bind port")
    parser.add_argument(
        "--backend",
        default="apiary",
        help=("Sandbox backend name or fully-qualified class path (default: apiary)"),
    )
    parser.add_argument(
        "--apiary-url",
        default=os.getenv("APIARY_URL", "http://127.0.0.1:8080"),
        help="Apiary daemon URL",
    )
    parser.add_argument(
        "--apiary-token",
        default=os.getenv("APIARY_API_TOKEN"),
        help="Apiary daemon bearer token",
    )
    parser.add_argument(
        "--mcp-token",
        default=os.getenv("MCP_AUTH_TOKEN"),
        help="Require this bearer token on the MCP SSE endpoint",
    )
    parser.add_argument(
        "--working-dir",
        default=os.getenv("APIARY_WORKING_DIR", "/workspace"),
        help="Default sandbox working directory",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=300.0,
        help="Seconds before an unconnected sandbox is reaped (default 300)",
    )
    parser.add_argument(
        "--transport",
        choices=["sse", "stdio"],
        default="sse",
        help="MCP transport (default: sse)",
    )
    parser.add_argument(
        "--backlog",
        type=int,
        default=2048,
        help="TCP listen backlog (default 2048)",
    )
    parser.add_argument(
        "--limit-concurrency",
        type=int,
        default=500,
        help="Max concurrent connections (default 500)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable ASGI debug mode (default: off)",
    )
    args = parser.parse_args()

    config = ServerConfig(
        backend=args.backend,
        backend_kwargs=_build_backend_kwargs(args),
        mcp_auth_token=args.mcp_token,
        debug=args.debug,
    )
    mcp, _backend, starlette_app = create_server(config)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        LOGGER.info(
            "Starting Coder MCP server on %s:%s (backend=%s)",
            args.host,
            args.port,
            args.backend,
        )
        uvicorn.run(
            starlette_app,
            host=args.host,
            port=args.port,
            backlog=args.backlog,
            limit_concurrency=args.limit_concurrency,
            timeout_keep_alive=30,
            log_level="info",
        )


if __name__ == "__main__":
    main()
