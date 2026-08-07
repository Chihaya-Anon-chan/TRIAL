"""Per-environment history buffers used to render multi-turn prompts."""

from __future__ import annotations

from typing import Any


class SimpleMemory:
    def reset(self, batch_size: int) -> None:
        self._rows: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]

    def __len__(self) -> int:
        return len(self._rows[0]) if self._rows else 0

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        return self._rows[index]

    def store(self, values: dict[str, list[Any]]) -> None:
        for index, row in enumerate(self._rows):
            row.append({key: value[index] for key, value in values.items()})

    def fetch(self, limit: int, obs_key: str, action_key: str) -> tuple[list[str], list[int]]:
        contexts, lengths = [], []
        for row in self._rows:
            entries = row[-limit:]
            contexts.append(
                "\n".join(
                    f"Observation: {entry.get(obs_key, '')}\nAction: {entry.get(action_key, '')}"
                    for entry in entries
                )
            )
            lengths.append(len(entries))
        return contexts, lengths


class SearchMemory(SimpleMemory):
    pass
