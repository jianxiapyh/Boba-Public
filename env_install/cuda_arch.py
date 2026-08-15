"""Resolve numeric CUDA architectures for Boba's extension build."""

from __future__ import annotations

import os
import re
from typing import Mapping, Optional, Sequence, Tuple


_ARCH_TOKEN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)(?P<ptx>\+PTX)?$")


def parse_arch_list(value: str) -> Tuple[str, ...]:
    tokens = [token for token in re.split(r"[;,\s]+", str(value).strip()) if token]
    if not tokens:
        raise RuntimeError("TORCH_CUDA_ARCH_LIST must not be empty.")

    architectures = {}
    for token in tokens:
        match = _ARCH_TOKEN.fullmatch(token)
        if match is None:
            raise RuntimeError(
                "TORCH_CUDA_ARCH_LIST must contain numeric capabilities such as "
                f"'8.9;12.0+PTX'; found {token!r}."
            )
        capability = (int(match.group("major")), int(match.group("minor")))
        architectures[capability] = architectures.get(capability, False) or bool(
            match.group("ptx")
        )

    return tuple(
        f"{major}.{minor}{'+PTX' if architectures[(major, minor)] else ''}"
        for major, minor in sorted(architectures)
    )


def detect_visible_architectures(torch_module) -> Tuple[str, ...]:
    try:
        device_count = int(torch_module.cuda.device_count())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Unable to inspect visible CUDA devices. Set TORCH_CUDA_ARCH_LIST "
            "explicitly for a headless build."
        ) from exc
    if device_count < 1:
        raise RuntimeError(
            "No CUDA GPU is visible. Set TORCH_CUDA_ARCH_LIST explicitly, for "
            "example TORCH_CUDA_ARCH_LIST='8.9;12.0'."
        )

    capabilities = set()
    for device_index in range(device_count):
        try:
            major, minor = torch_module.cuda.get_device_capability(device_index)
            capabilities.add((int(major), int(minor)))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unable to read CUDA capability for device {device_index}."
            ) from exc
    return tuple(f"{major}.{minor}" for major, minor in sorted(capabilities))


def resolve_architectures(
    torch_module=None,
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[str, ...]:
    source = os.environ if environ is None else environ
    requested = source.get("TORCH_CUDA_ARCH_LIST", "")
    if str(requested).strip():
        return parse_arch_list(requested)

    if torch_module is None:
        import torch as torch_module

    return detect_visible_architectures(torch_module)


def expected_sm_targets(architectures: Sequence[str]) -> Tuple[str, ...]:
    normalized = parse_arch_list(";".join(architectures))
    return tuple(
        "sm_" + architecture.removesuffix("+PTX").replace(".", "")
        for architecture in normalized
    )


def main() -> None:
    try:
        architectures = resolve_architectures()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(";".join(architectures))


if __name__ == "__main__":
    main()
