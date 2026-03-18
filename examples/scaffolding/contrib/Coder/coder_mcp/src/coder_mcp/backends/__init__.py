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
"""Sandbox backend abstraction: Protocol, registry, and factory."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

_BACKEND_MAPPING: dict[str, str] = {
    "apiary": "coder_mcp.backends.apiary.ApiarySandbox",
}


@dataclass
class ExecutionResult:
    """Normalized result from running a command in any sandbox backend."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


@runtime_checkable
class SandboxBackend(Protocol):
    """Abstraction over sandbox execution environments.

    Implementations manage per-client sandbox sessions and provide command
    execution.  The server layer calls ``attach`` / ``detach`` to track
    SSE connection lifecycles and ``execute`` to run commands.
    """

    @property
    def working_dir(self) -> str: ...

    @property
    def active_sessions(self) -> int: ...

    def attach(self, client_id: str) -> None: ...

    def detach(self, client_id: str) -> None: ...

    def start_reaper(self) -> None: ...

    async def configure_session(
        self,
        client_id: str,
        *,
        base_image: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> None:
        """Pre-configure the sandbox for *client_id* before the first command.

        Parameters
        ----------
        client_id:
            The client whose session should be configured.
        base_image:
            Local rootfs path (already on disk) to use as the sandbox
            filesystem.  When set, the Apiary daemon creates a dedicated
            sandbox with this rootfs instead of using the pool.
        working_dir:
            Override the default working directory for this session.
        """
        ...

    async def execute(
        self,
        command: str,
        *,
        client_id: str,
        timeout_ms: Optional[int] = None,
        working_dir: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ExecutionResult: ...

    async def shutdown(self) -> None: ...


def get_backend_class(spec: str) -> type:
    full_path = _BACKEND_MAPPING.get(spec, spec)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as exc:
        available = ", ".join(sorted(_BACKEND_MAPPING))
        raise ValueError(
            f"Unknown backend: {spec!r} (resolved to {full_path!r}, available: {available})"
        ) from exc


def get_backend(backend: str, **kwargs) -> SandboxBackend:
    """Instantiate a sandbox backend by name with configuration kwargs."""
    cls = get_backend_class(backend)
    return cls(**kwargs)
