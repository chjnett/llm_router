"""Guarded stdlib-only child process for one MBPP candidate."""

from __future__ import annotations

import ast
import builtins
import json
import sys


ALLOWED_IMPORTS = {
    "array", "bisect", "cmath", "collections", "copy", "datetime", "decimal", "fractions", "functools",
    "heapq", "itertools", "math", "operator", "random", "re", "statistics", "string",
    "sys", "typing",
}
BLOCKED_NAMES = {
    "breakpoint", "compile", "eval", "exec", "globals", "input", "locals", "open",
    "setattr", "vars", "memoryview",
}
BLOCKED_ATTRIBUTES = {
    "argv", "executable", "exit", "modules", "path", "setprofile", "settrace",
    "stderr", "stdin", "stdout",
}


def validate_candidate(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
            if any(name not in ALLOWED_IMPORTS for name in names):
                raise PermissionError(f"blocked import: {names}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if node.level or root not in ALLOWED_IMPORTS:
                raise PermissionError(f"blocked import: {node.module}")
        elif isinstance(node, ast.Name) and (
            node.id in BLOCKED_NAMES or (node.id.startswith("__") and node.id != "__name__")
        ):
            raise PermissionError(f"blocked name: {node.id}")
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("__") or node.attr in BLOCKED_ATTRIBUTES
        ):
            raise PermissionError(f"blocked attribute: {node.attr}")


def safe_import(name, globals_=None, locals_=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if level or root not in ALLOWED_IMPORTS:
        raise PermissionError(f"blocked import: {name}")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def main() -> None:
    payload = json.loads(sys.stdin.read())
    source = str(payload["code"])
    validate_candidate(source)
    safe_builtins = {
        name: getattr(builtins, name)
        for name in (
            "abs", "all", "any", "bin", "bool", "bytes", "chr", "complex", "dict", "divmod",
            "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "int",
            "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next",
            "object", "oct", "ord", "pow", "print", "range", "repr", "reversed", "round",
            "set", "slice", "sorted", "str", "sum", "super", "tuple", "type", "zip",
            "ArithmeticError", "AssertionError", "Exception", "IndexError", "KeyError",
            "RuntimeError", "StopIteration", "TypeError", "ValueError", "ZeroDivisionError",
        )
    }
    safe_builtins["__import__"] = safe_import
    namespace = {"__builtins__": safe_builtins, "__name__": "__mbpp_candidate__"}
    for statement in payload.get("test_imports", []):
        exec(statement, namespace, namespace)
    exec(compile(source, "<candidate>", "exec"), namespace, namespace)
    for test in payload["tests"]:
        exec(compile(test, "<test>", "exec"), namespace, namespace)
    print(json.dumps({"passed": True}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(json.dumps({"passed": False, "error": type(error).__name__, "message": str(error)[:300]}))
        raise SystemExit(1)
