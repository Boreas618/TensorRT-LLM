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
"""Per-client plan state management (local, no sandbox interaction)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlanState:
    """Tracks the current plan and explanation for a single client."""

    current_plan: list[dict[str, str]] = field(default_factory=list)
    explanation: Optional[str] = None


class PlanStore:
    """Thread-safe container for per-client :class:`PlanState` instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, PlanState] = {}

    def get(self, client_id: str) -> PlanState:
        """Return the plan state for *client_id*, creating if absent."""
        with self._lock:
            if client_id not in self._states:
                self._states[client_id] = PlanState()
            return self._states[client_id]

    def remove(self, client_id: str) -> None:
        """Remove the plan state for *client_id* (no-op if absent)."""
        with self._lock:
            self._states.pop(client_id, None)
