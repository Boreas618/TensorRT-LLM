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
"""Composition root: wires backends, tools, plan store, and ASGI app."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Optional

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

from coder_mcp.app import create_app
from coder_mcp.backends import SandboxBackend, get_backend
from coder_mcp.plan import PlanStore
from coder_mcp.tools import register_tools


@dataclass
class ServerConfig:
    """All settings needed to construct and run the MCP server."""

    # Backend selection
    backend: str = "apiary"
    backend_kwargs: dict = field(default_factory=dict)

    # MCP auth
    mcp_auth_token: Optional[str] = None

    # ASGI debug mode
    debug: bool = False


def create_server(config: ServerConfig) -> tuple[FastMCP, SandboxBackend, Starlette]:
    """Wire all components and return ``(mcp, backend, starlette_app)``.

    The caller is responsible for running the Starlette app (via uvicorn)
    or invoking ``mcp.run(transport="stdio")`` for stdio mode.
    """
    plan_store = PlanStore()

    backend_kwargs = dict(config.backend_kwargs)
    backend_kwargs.setdefault("on_client_destroy", lambda cid: plan_store.remove(cid))
    sandbox: SandboxBackend = get_backend(config.backend, **backend_kwargs)

    client_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
        "client_id", default="stdio"
    )

    mcp = FastMCP("coder_mcp")
    register_tools(mcp, sandbox, plan_store, client_id_var)

    app = create_app(
        mcp._mcp_server,
        sandbox,
        client_id_var,
        auth_token=config.mcp_auth_token,
        debug=config.debug,
    )

    return mcp, sandbox, app
