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
"""Apiary sandbox backend.

Manages per-client Apiary sessions and executes commands via the Apiary
daemon's REST API (``/api/v1/sessions``, ``/api/v1/tasks``).
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from typing import Callable, Optional

import httpx

from coder_mcp.backends import ExecutionResult

LOGGER = logging.getLogger(__name__)


def _raise_with_body(resp: httpx.Response) -> None:
    """Call *raise_for_status* but include the response body in the error."""
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = resp.text[:2000]
        LOGGER.error(
            "Apiary %s %s → %d\n%s",
            resp.request.method,
            resp.request.url,
            resp.status_code,
            body,
        )
        raise httpx.HTTPStatusError(
            f"{exc.args[0]}\nResponse body:\n{body}",
            request=exc.request,
            response=exc.response,
        ) from None


_REAPER_INTERVAL = 60


class ApiarySandbox:
    """Apiary-backed :class:`SandboxBackend` implementation.

    Each ``client_id`` maps to one Apiary session.  Sandboxes outlive SSE
    connections and are reaped after *idle_timeout* seconds of inactivity.
    """

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        working_dir: str = "/workspace",
        idle_timeout: float = 1800.0,
        on_client_destroy: Optional[Callable[[str], None]] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._working_dir = working_dir
        self._idle_timeout = idle_timeout
        self._on_client_destroy = on_client_destroy

        self._sessions: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._refcounts: dict[str, int] = {}
        self._detached_at: dict[str, float] = {}
        self._session_configs: dict[str, dict] = {}

        self._client: Optional[httpx.AsyncClient] = None
        self._reaper_task: Optional[asyncio.Task] = None

    @property
    def working_dir(self) -> str:
        return self._working_dir

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(timeout=300.0),
                limits=httpx.Limits(
                    max_connections=500,
                    max_keepalive_connections=200,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    def _lock_for(self, cid: str) -> asyncio.Lock:
        if cid not in self._locks:
            self._locks[cid] = asyncio.Lock()
        return self._locks[cid]

    async def configure_session(
        self,
        client_id: str,
        *,
        base_image: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> None:
        """Pre-configure the sandbox for *client_id*.

        Must be called **before** the first command execution.  The
        *base_image* must be a local rootfs path already on disk (exported
        by :class:`~apiary_swebench.rootfs.RootfsManager` at init time).
        """
        cfg: dict = {}
        if base_image is not None:
            cfg["base_image"] = base_image
        if working_dir is not None:
            cfg["working_dir"] = working_dir
        elif base_image is not None:
            cfg["working_dir"] = "/testbed"
        if cfg:
            self._session_configs[client_id] = cfg
            LOGGER.info("Session config set for client %s: %s", client_id, cfg)

    async def _create_apiary_session(self, cid: str) -> str:
        client = await self._get_client()
        cfg = self._session_configs.get(cid, {})
        payload: dict = {
            "working_dir": cfg.get("working_dir", self._working_dir),
        }
        if "base_image" in cfg:
            payload["base_image"] = cfg["base_image"]
        resp = await client.post("/api/v1/sessions", json=payload)
        _raise_with_body(resp)
        return resp.json()["session_id"]

    async def _destroy_apiary_session(self, session_id: str) -> None:
        try:
            client = await self._get_client()
            await client.delete(f"/api/v1/sessions/{session_id}")
        except Exception:
            LOGGER.warning("Failed to destroy session %s", session_id, exc_info=True)

    async def _ensure_session(self, cid: str) -> str:
        lock = self._lock_for(cid)
        async with lock:
            if cid in self._sessions:
                return self._sessions[cid]
            session_id = await self._create_apiary_session(cid)
            self._sessions[cid] = session_id
            LOGGER.info(
                "Session %s created for client %s (%d active)",
                session_id,
                cid,
                len(self._sessions),
            )
            return session_id

    async def _destroy_client(self, cid: str) -> None:
        session_id = self._sessions.pop(cid, None)
        self._locks.pop(cid, None)
        self._refcounts.pop(cid, None)
        self._detached_at.pop(cid, None)
        self._session_configs.pop(cid, None)
        if self._on_client_destroy:
            self._on_client_destroy(cid)
        if session_id:
            await self._destroy_apiary_session(session_id)
            LOGGER.info(
                "Session %s destroyed for client %s (%d remaining)",
                session_id,
                cid,
                len(self._sessions),
            )

    def attach(self, client_id: str) -> None:
        self._refcounts[client_id] = self._refcounts.get(client_id, 0) + 1
        self._detached_at.pop(client_id, None)

    def detach(self, client_id: str) -> None:
        count = self._refcounts.get(client_id, 1) - 1
        if count <= 0:
            self._refcounts.pop(client_id, None)
            self._detached_at[client_id] = time.monotonic()
        else:
            self._refcounts[client_id] = count

    def start_reaper(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reap_loop())

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(_REAPER_INTERVAL)
            now = time.monotonic()
            for cid in list(self._detached_at):
                if cid in self._refcounts:
                    self._detached_at.pop(cid, None)
                    continue
                if cid not in self._sessions:
                    self._detached_at.pop(cid, None)
                    continue
                if now - self._detached_at[cid] >= self._idle_timeout:
                    LOGGER.info("Reaping idle client %s", cid)
                    await self._destroy_client(cid)

    async def execute(
        self,
        command: str,
        *,
        client_id: str,
        timeout_ms: Optional[int] = None,
        working_dir: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ExecutionResult:
        """Run *command* in the sandbox for *client_id*."""
        wrapped = f"/bin/sh -c {shlex.quote(command)}"
        session_id = await self._ensure_session(client_id)
        client = await self._get_client()

        payload: dict = {"command": wrapped, "session_id": session_id}
        if timeout_ms is not None:
            payload["timeout_ms"] = timeout_ms
        if working_dir is not None:
            payload["working_dir"] = working_dir
        if env:
            payload["env"] = env

        resp = await client.post("/api/v1/tasks", json=payload)

        if resp.status_code == 404:
            LOGGER.warning(
                "Session %s lost for client %s, recreating…",
                session_id,
                client_id,
            )
            lock = self._lock_for(client_id)
            async with lock:
                self._sessions.pop(client_id, None)
            session_id = await self._ensure_session(client_id)
            payload["session_id"] = session_id
            resp = await client.post("/api/v1/tasks", json=payload)

        _raise_with_body(resp)
        data = resp.json()
        return ExecutionResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=data.get("exit_code", -1),
            timed_out=bool(data.get("timed_out")),
        )

    async def shutdown(self) -> None:
        if self._reaper_task:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
        for cid in list(self._sessions):
            await self._destroy_client(cid)
        if self._client and not self._client.is_closed:
            await self._client.aclose()
