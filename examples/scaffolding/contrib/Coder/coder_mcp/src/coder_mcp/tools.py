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
"""MCP tool definitions for the Coder agent.

All tools are registered via :func:`register_tools` against a caller-provided
:class:`~mcp.server.fastmcp.FastMCP` instance, keeping this module free of
module-level global state.
"""

from __future__ import annotations

import base64
import contextvars
import re
import shlex
from typing import Optional

from mcp.server.fastmcp import FastMCP

from coder_mcp.backends import ExecutionResult, SandboxBackend
from coder_mcp.patch import apply_hunks, parse_patch
from coder_mcp.plan import PlanStore

_Q = shlex.quote


def _fmt_result(result: ExecutionResult) -> str:
    """Format an :class:`ExecutionResult` into the string the agent expects."""
    parts: list[str] = []
    stdout = result.stdout.rstrip("\n")
    stderr = result.stderr.rstrip("\n")
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(stderr)
    if result.timed_out:
        parts.append("[timed out]")
    body = "\n".join(parts) if parts else "(no output)"
    return f"{body}\n\n[Exit code: {result.exit_code}]"


def register_tools(
    mcp: FastMCP,
    backend: SandboxBackend,
    plan_store: PlanStore,
    client_id_var: contextvars.ContextVar[str],
) -> None:
    """Register all Coder MCP tools on *mcp*.

    Tools obtain the current client identity from *client_id_var* and
    delegate execution to *backend*.
    """

    def _cid() -> str:
        return client_id_var.get()

    @mcp.tool()
    async def read_file(
        file_path: str,
        offset: int = 1,
        limit: Optional[int] = None,
        mode: str = "slice",
    ) -> str:
        """Read a file with 1-indexed line numbers from the sandbox.

        Args:
            file_path: Absolute path to the file.
            offset: Line number to start reading from (1-indexed, default 1).
            limit: Maximum number of lines to return.
            mode: "slice" for simple line ranges (default).
        """
        _ = mode
        cid = _cid()
        try:
            wc = await backend.execute(f"wc -l < {_Q(file_path)}", client_id=cid)
            total = 0
            if wc.exit_code == 0:
                try:
                    total = int(wc.stdout.strip())
                except ValueError:
                    pass

            cond = f"NR>={offset}"
            if limit is not None:
                cond += f" && NR<{offset + limit}"
            awk_cmd = f"awk '{cond} {{printf \"%6d|\", NR; print}}' {_Q(file_path)}"
            result = await backend.execute(awk_cmd, client_id=cid)
            if result.exit_code != 0:
                err = (result.stderr or result.stdout).strip()
                return f"Error: {err}"

            content = result.stdout
            returned = content.count("\n") if content else 0
            return f"{content}\n[Total lines: {total}, Lines returned: {returned}]"
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    async def list_dir(
        dir_path: str,
        offset: int = 1,
        limit: Optional[int] = None,
        depth: int = 1,
    ) -> str:
        """List directory contents with type labels and 1-indexed entry numbers.

        Directories are shown first, then files, sorted alphabetically.

        Args:
            dir_path: Absolute path to the directory.
            offset: Entry number to start listing from (1-indexed, default 1).
            limit: Maximum entries to return.
            depth: Maximum directory depth to traverse (default 1).
        """
        cid = _cid()
        try:
            cmd = (
                f"find {_Q(dir_path)} -maxdepth {int(depth)} -mindepth 1"
                f" -printf '%y %P\\n' 2>/dev/null | sort"
            )
            result = await backend.execute(cmd, client_id=cid)
            if result.exit_code != 0:
                err = result.stderr.strip()
                return f"Error: {err or 'find failed'}"

            raw = result.stdout.strip()
            if not raw:
                return "(empty directory)\n\n[Total entries: 0, Entries returned: 0]"

            entries: list[tuple[str, str]] = []
            for line in raw.split("\n"):
                if not line or len(line) < 3:
                    continue
                type_ch, name = line[0], line[2:]
                entries.append(("dir" if type_ch == "d" else "file", name))

            entries.sort(key=lambda e: (0 if e[0] == "dir" else 1, e[1]))
            total = len(entries)
            start = offset - 1
            selected = entries[start : start + limit] if limit else entries[start:]

            out_lines = [
                f"{offset + i:6d}. [{etype:5s}] {name}" for i, (etype, name) in enumerate(selected)
            ]
            returned = len(selected)
            return (
                "\n".join(out_lines) + f"\n\n[Total entries: {total}, Entries returned: {returned}]"
            )
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    async def grep_files(
        pattern: str,
        include: Optional[str] = None,
        path: Optional[str] = None,
        limit: int = 100,
    ) -> str:
        """Search files for a regex pattern inside the sandbox using ripgrep.

        Args:
            pattern: Regular expression pattern to search for.
            include: Glob pattern to filter files (e.g. "*.py").
            path: Directory or file to search (default: working directory).
            limit: Max file paths to return (default 100).
        """
        cid = _cid()
        try:
            parts = [
                "rg",
                "--line-number",
                "--no-heading",
                "--max-filesize",
                "1M",
                "--max-count",
                "10",
            ]
            if include:
                parts.extend(["--glob", _Q(include)])
            parts.append("--")
            parts.append(_Q(pattern))
            if path:
                parts.append(_Q(path))

            result = await backend.execute(" ".join(parts), client_id=cid, timeout_ms=30000)
            stdout = result.stdout
            if result.exit_code not in (0, 1) and not stdout:
                err = result.stderr.strip()
                return f"Error: {err or 'rg failed'}"
            if not stdout.strip():
                return "No matches found.\n\n[Files matched: 0]"

            files: dict[str, list[tuple[int, str]]] = {}
            for line in stdout.strip().split("\n"):
                m = re.match(r"^(.+?):(\d+):(.*)", line)
                if m:
                    fp, ln, ct = m.group(1), int(m.group(2)), m.group(3)
                    files.setdefault(fp, []).append((ln, ct))

            out: list[str] = []
            file_count = 0
            for fp, matches in files.items():
                if file_count >= limit:
                    break
                file_count += 1
                out.append(f"\n{fp}:")
                for ln, ct in matches[:10]:
                    out.append(f"  {ln}: {ct[:200]}{'...' if len(ct) > 200 else ''}")
                if len(matches) > 10:
                    out.append(f"  ... ({len(matches) - 10} more matches)")
            body = "\n".join(out).strip()
            return f"{body}\n\n[Files matched: {file_count}]"
        except Exception as exc:
            return f"Error: {exc}"

    @mcp.tool()
    async def apply_patch(patch: str) -> str:
        """Apply a Codex-style structured patch inside the sandbox.

        Supports ``*** Add File``, ``*** Delete File``, and
        ``*** Update File`` operations wrapped in a
        ``*** Begin Patch`` / ``*** End Patch`` envelope.

        Args:
            patch: The patch content.
        """
        cid = _cid()
        try:
            ops = parse_patch(patch)
        except Exception as exc:
            return f"Error parsing patch: {exc}"
        if not ops:
            return "Error: no file operations found in patch"

        results: list[str] = []
        for op in ops:
            try:
                if op.kind == "add":
                    content = "\n".join(op.content_lines)
                    encoded = base64.b64encode(content.encode()).decode()
                    cmd = (
                        f'mkdir -p -- "$(dirname {_Q(op.path)})" && '
                        f"printf '%s' {_Q(encoded)} | base64 -d > {_Q(op.path)}"
                    )
                    r = await backend.execute(cmd, client_id=cid)
                    if r.exit_code != 0:
                        results.append(f"Add {op.path}: FAILED - {(r.stderr or r.stdout).strip()}")
                    else:
                        results.append(f"Add {op.path}: OK")

                elif op.kind == "delete":
                    r = await backend.execute(f"rm -f -- {_Q(op.path)}", client_id=cid)
                    if r.exit_code != 0:
                        results.append(f"Delete {op.path}: FAILED - {r.stderr.strip()}")
                    else:
                        results.append(f"Delete {op.path}: OK")

                elif op.kind == "update":
                    r = await backend.execute(f"cat -- {_Q(op.path)}", client_id=cid)
                    if r.exit_code != 0:
                        results.append(f"Update {op.path}: FAILED (read) - {r.stderr.strip()}")
                        continue

                    updated = apply_hunks(r.stdout, op.hunks)
                    target = op.move_to or op.path
                    encoded = base64.b64encode(updated.encode()).decode()
                    w = await backend.execute(
                        f'mkdir -p -- "$(dirname {_Q(target)})" && '
                        f"printf '%s' {_Q(encoded)} | base64 -d > {_Q(target)}",
                        client_id=cid,
                    )
                    if w.exit_code != 0:
                        results.append(f"Update {op.path}: FAILED (write) - {w.stderr.strip()}")
                        continue
                    if op.move_to and op.move_to != op.path:
                        await backend.execute(f"rm -f -- {_Q(op.path)}", client_id=cid)

                    label = f"Update {op.path}"
                    if op.move_to:
                        label += f" -> {op.move_to}"
                    results.append(f"{label}: OK")

            except Exception as exc:
                results.append(f"{op.kind.title()} {op.path}: ERROR - {exc}")

        return "; ".join(results) if results else "No changes applied"

    @mcp.tool()
    async def exec(
        command: list[str],
        workdir: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> str:
        """Execute a command array directly via execvp (no shell interpretation).

        Args:
            command: Command array, e.g. ["ls", "-la", "/workspace"].
            workdir: Working directory for execution.
            timeout_ms: Timeout in milliseconds.
        """
        if not command:
            return "Error: Command array cannot be empty"
        result = await backend.execute(
            shlex.join(command),
            client_id=_cid(),
            timeout_ms=timeout_ms,
            working_dir=workdir,
        )
        return _fmt_result(result)

    @mcp.tool()
    async def shell(
        command: str,
        workdir: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> str:
        """Run a shell command string inside the sandbox (pipes, redirects, etc. work).

        Args:
            command: Shell command string.
            workdir: Working directory for execution.
            timeout_ms: Timeout in milliseconds.
        """
        if not command:
            return "Error: Command cannot be empty"
        result = await backend.execute(
            command,
            client_id=_cid(),
            timeout_ms=timeout_ms,
            working_dir=workdir,
        )
        return _fmt_result(result)

    @mcp.tool()
    async def update_plan(
        plan: list[dict[str, str]],
        explanation: Optional[str] = None,
    ) -> str:
        """Update the task plan with steps and progress tracking.

        At most one step can be ``in_progress`` at a time.

        Args:
            plan: List of ``{"step": "...", "status": "pending|in_progress|completed"}``.
            explanation: Optional explanation for the plan change.
        """
        if not plan:
            return "Error: plan is required and cannot be empty"

        valid = {"pending", "in_progress", "completed"}
        ip_count = 0
        validated: list[dict[str, str]] = []
        for idx, item in enumerate(plan, 1):
            if not isinstance(item, dict):
                return f"Error: Plan item {idx} must be an object"
            step, status = item.get("step"), item.get("status")
            if not step:
                return f"Error: Plan item {idx} missing 'step' field"
            if not status:
                return f"Error: Plan item {idx} missing 'status' field"
            if status not in valid:
                return (
                    f"Error: Plan item {idx} has invalid status '{status}'. "
                    f"Must be one of: {', '.join(sorted(valid))}"
                )
            if status == "in_progress":
                ip_count += 1
            validated.append({"step": step, "status": status})

        if ip_count > 1:
            return f"Error: At most one step can be in_progress, found {ip_count}"

        st = plan_store.get(_cid())
        st.current_plan = validated
        st.explanation = explanation

        out: list[str] = []
        if explanation:
            out += [f"Explanation: {explanation}", ""]
        out.append("Plan:")
        for idx, it in enumerate(validated, 1):
            out.append(f"  {idx}. [{it['status']}] {it['step']}")
        total = len(validated)
        done = sum(1 for it in validated if it["status"] == "completed")
        pct = round(done / total * 100, 1) if total else 0
        out += ["", f"Progress: {done}/{total} ({pct}%)"]
        return "\n".join(out)

    @mcp.tool()
    async def think(thought: str) -> str:
        """Record an internal thought about the current task (no action executed).

        Args:
            thought: The thought or reflection text.
        """
        return f"Thought recorded: {thought}"

    @mcp.tool()
    async def complete_task(summary: str) -> str:
        """Signal that the current task is complete.

        Args:
            summary: Brief summary of what was accomplished.
        """
        return f"Task completed: {summary}"
