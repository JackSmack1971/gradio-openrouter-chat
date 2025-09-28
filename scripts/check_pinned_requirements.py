"""Pre-commit hook to ensure requirements are strictly pinned."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Security: enforce exact version pinning to prevent supply-chain surprises.
PINNED_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?=="
    r"[A-Za-z0-9_.-]+(?:\\.[A-Za-z0-9_.-]+)*$"
)


def is_pinned(line: str) -> bool:
    """Check if a requirement line is pinned exactly."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    if stripped.startswith("-r ") or stripped.startswith("--"):
        # Security: do not allow nested requirements or pip options in this file.
        return False
    return bool(PINNED_PATTERN.match(stripped.split(";")[0]))


def main(path: str = "requirements.txt") -> int:
    requirements_path = Path(path)
    if not requirements_path.exists():
        print(f"requirements file not found: {requirements_path}", file=sys.stderr)
        return 1

    violations: list[str] = []
    for number, raw_line in enumerate(
        requirements_path.read_text().splitlines(), start=1
    ):
        if not is_pinned(raw_line):
            violations.append(
                f"Line {number}: '{raw_line.strip() or '<empty>'}' is not an exact pin"
            )

    if violations:
        print(
            "Found unpinned or unsupported specifiers in requirements.txt:",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        print(
            "Please pin dependencies using 'pip-compile' or manual == specifiers.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
