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
"""Codex-style structured patch parser and applicator.

Handles ``*** Begin Patch`` / ``*** End Patch`` envelopes with
``Add File``, ``Delete File``, and ``Update File`` operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class PatchOp:
    """A single file operation parsed from a Codex patch."""

    kind: str  # "add" | "delete" | "update"
    path: str
    move_to: Optional[str] = None
    content_lines: list[str] = field(default_factory=list)
    hunks: list[dict[str, Any]] = field(default_factory=list)


def parse_patch(text: str) -> list[PatchOp]:
    """Parse a ``*** Begin Patch`` / ``*** End Patch`` envelope."""
    lines = text.split("\n")
    ops: list[PatchOp] = []
    i = 0
    while i < len(lines) and lines[i].strip() != "*** Begin Patch":
        i += 1
    i += 1

    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "*** End Patch":
            break

        if stripped.startswith("*** Add File:"):
            path = stripped[len("*** Add File:") :].strip()
            content: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("***"):
                raw = lines[i]
                content.append(raw[1:] if raw.startswith("+") else raw)
                i += 1
            ops.append(PatchOp(kind="add", path=path, content_lines=content))
            continue

        if stripped.startswith("*** Delete File:"):
            path = stripped[len("*** Delete File:") :].strip()
            ops.append(PatchOp(kind="delete", path=path))
            i += 1
            continue

        if stripped.startswith("*** Update File:"):
            path = stripped[len("*** Update File:") :].strip()
            move_to: Optional[str] = None
            hunks: list[dict[str, Any]] = []
            cur: Optional[dict[str, Any]] = None
            i += 1
            while i < len(lines):
                raw = lines[i]
                s = raw.strip()
                if s.startswith("***") and not s.startswith("*** Move to:"):
                    break
                if s.startswith("*** Move to:"):
                    move_to = s[len("*** Move to:") :].strip()
                elif raw.startswith("@@"):
                    anchor = raw[2:].strip() if len(raw) > 2 else ""
                    cur = {"anchor": anchor, "changes": []}
                    hunks.append(cur)
                elif cur is not None:
                    if raw.startswith("-"):
                        cur["changes"].append(("-", raw[1:]))
                    elif raw.startswith("+"):
                        cur["changes"].append(("+", raw[1:]))
                    elif raw.startswith(" "):
                        cur["changes"].append((" ", raw[1:]))
                    else:
                        cur["changes"].append((" ", raw))
                i += 1
            ops.append(PatchOp(kind="update", path=path, move_to=move_to, hunks=hunks))
            continue

        i += 1
    return ops


def apply_hunks(content: str, hunks: list[dict[str, Any]]) -> str:
    """Apply parsed hunks to file content."""
    lines = content.split("\n")
    for hunk in hunks:
        anchor: str = hunk["anchor"]
        changes: list[tuple[str, str]] = hunk["changes"]
        if not anchor and not changes:
            continue

        anchor_idx: Optional[int] = None
        if anchor:
            stripped_anchor = anchor.rstrip()
            for idx in range(len(lines)):
                if lines[idx].rstrip() == stripped_anchor:
                    anchor_idx = idx
                    break
            if anchor_idx is None:
                for idx in range(len(lines)):
                    if stripped_anchor in lines[idx]:
                        anchor_idx = idx
                        break
            if anchor_idx is None:
                LOGGER.warning("Anchor not found: %r", anchor)
                continue
        else:
            anchor_idx = 0

        read_pos = anchor_idx
        new_section: list[str] = []
        for op, text in changes:
            if op == " ":
                if read_pos < len(lines):
                    new_section.append(lines[read_pos])
                    read_pos += 1
            elif op == "-":
                if read_pos < len(lines):
                    read_pos += 1
            elif op == "+":
                new_section.append(text)
        lines[anchor_idx:read_pos] = new_section
    return "\n".join(lines)
