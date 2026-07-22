from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class AcceleratorBackend:
    name: str
    device_type: str
    module: Any

    def is_available(self) -> bool:
        return bool(self.module.is_available())

    def normalize_device(self, device: Any) -> tuple[str, int | None]:
        value = torch.device(device)
        if value.type != self.device_type:
            raise ValueError(f"expected a {self.device_type} device, got {value}")
        return ("accelerator", value.index)


def compare_observations(observations):
    if not observations:
        raise ValueError("observations must contain at least one backend")

    ordered = sorted(observations.items())
    expected_fields = set(ordered[0][1])
    if any(set(observation) != expected_fields for _, observation in ordered[1:]):
        raise ValueError("backend observations must contain the same fields")

    differences = {}
    for field in sorted(expected_fields):
        values = {name: observation[field] for name, observation in ordered}
        first_value = next(iter(values.values()))
        if any(value != first_value for value in values.values()):
            differences[field] = values
    return differences
