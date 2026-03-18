# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Starlette ASGI application factory with SSE transport."""

from __future__ import annotations

import contextvars
import hmac
import logging
import uuid
from typing import Optional

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from coder_mcp.backends import SandboxBackend

LOGGER = logging.getLogger(__name__)


def create_app(
    mcp_server: Server,
    backend: SandboxBackend,
    client_id_var: contextvars.ContextVar[str],
    auth_token: Optional[str] = None,
    *,
    debug: bool = False,
) -> Starlette:
    """Build a Starlette ASGI app that serves MCP over SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        if auth_token:
            provided = request.headers.get("authorization", "")
            expected = f"Bearer {auth_token}"
            if not hmac.compare_digest(provided.encode(), expected.encode()):
                return JSONResponse({"error": "unauthorized"}, status_code=401)

        cid = request.query_params.get("client_id") or uuid.uuid4().hex[:12]
        client_id_var.set(cid)
        backend.attach(cid)

        base_image = request.query_params.get("base_image")
        if base_image:
            await backend.configure_session(cid, base_image=base_image)

        LOGGER.info("Client %s connected (base_image=%s)", cid, base_image)

        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            try:
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )
            finally:
                backend.detach(cid)
                LOGGER.info("Client %s disconnected", cid)
        return Response()

    async def handle_health(_: Request):
        return JSONResponse({"status": "ok", "sessions": backend.active_sessions})

    async def on_startup():
        backend.start_reaper()

    async def on_shutdown():
        await backend.shutdown()

    return Starlette(
        debug=debug,
        routes=[
            Route("/health", endpoint=handle_health, methods=["GET"]),
            Route("/sse", endpoint=handle_sse, methods=["GET"]),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        on_startup=[on_startup],
        on_shutdown=[on_shutdown],
    )
