from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path("src/umutextstats")


def method_calls(
    function: ast.FunctionDef,
) -> set[str]:
    calls = set()

    for node in ast.walk(
        function
    ):
        if isinstance(
            node,
            ast.Call,
        ):
            target = node.func

            if isinstance(
                target,
                ast.Attribute,
            ):
                calls.add(
                    target.attr
                )

            elif isinstance(
                target,
                ast.Name,
            ):
                calls.add(
                    target.id
                )

    return calls


for path in ROOT.rglob("*.py"):
    source = path.read_text(
        encoding="utf-8"
    )

    try:
        tree = ast.parse(
            source
        )
    except SyntaxError:
        continue

    for node in tree.body:
        if not isinstance(
            node,
            ast.ClassDef,
        ):
            continue

        methods = {
            item.name: item
            for item in node.body
            if isinstance(
                item,
                ast.FunctionDef,
            )
        }

        if (
            "inspect" not in methods
            and "compute_result" not in methods
        ):
            continue

        inspect_calls = (
            method_calls(
                methods["inspect"]
            )
            if "inspect" in methods
            else set()
        )

        compute_calls = (
            method_calls(
                methods["compute_result"]
            )
            if "compute_result" in methods
            else set()
        )

        shared = sorted(
            inspect_calls
            & compute_calls
        )

        print()
        print(
            f"{path}:{node.lineno}"
        )
        print(
            f"Class: {node.name}"
        )
        print(
            "Has inspect:",
            "inspect" in methods,
        )
        print(
            "Has compute_result:",
            "compute_result" in methods,
        )
        print(
            "Shared calls:",
            ", ".join(shared)
            if shared
            else "(none)",
        )
        print(
            "Inspect-only calls:",
            ", ".join(
                sorted(
                    inspect_calls
                    - compute_calls
                )
            )
            or "(none)",
        )
        print(
            "Compute-only calls:",
            ", ".join(
                sorted(
                    compute_calls
                    - inspect_calls
                )
            )
            or "(none)",
        )