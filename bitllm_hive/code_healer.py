"""Code healer: turn broken or noisy generated code into parseable Python."""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Any


class CodeHealer:
    """A minimal code repair agent.

    It attempts to clean common generation artefacts and re-parse the code.
    A real healer can plug in a separate LLM call, a linter, or symbolic
    reasoning; this scaffold focuses on syntax-level first-aid.
    """

    def __init__(self, max_attempts: int = 5):
        self.max_attempts = max_attempts

    def heal(self, code: str) -> tuple[str, list[str]]:
        """Return ``(healed_code, log_of_fixes)``."""
        original = code
        fixes: list[str] = []
        current = code

        for attempt in range(self.max_attempts):
            try:
                ast.parse(current)
                if attempt > 0:
                    fixes.append(f"parseable after {attempt} healing attempt(s)")
                return current, fixes
            except SyntaxError as exc:
                next_version = self._first_aid(current, exc)
                if next_version == current:
                    break
                fixes.append(self._describe_fix(exc))
                current = next_version

        if current == original:
            fixes.append("could not heal: no applicable rules")
        else:
            fixes.append("healing incomplete: still unparseable")
        return current, fixes

    def _first_aid(self, code: str, exc: SyntaxError) -> str:
        lines = code.splitlines(keepends=True)

        # 1. Strip markdown fences and REPL prompts.
        cleaned = re.sub(r"```python\s*|```\s*", "", code, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(>>>|\.\.\.)\s*", "", cleaned, flags=re.MULTILINE)

        # 2. Fix missing final newline.
        if cleaned and not cleaned.endswith("\n"):
            cleaned += "\n"

        # 3. Fix missing colon before indented block.
        match = re.search(r"line\s+(\d+)", str(exc.msg))
        try:
            lineno = int(match.group(1)) if match else exc.lineno
        except Exception:
            lineno = exc.lineno

        if lineno and 1 <= lineno <= len(lines):
            target = lines[lineno - 1].rstrip()
            if target and not target.rstrip().endswith(":"):
                keywords = ("if", "else", "elif", "for", "while", "def", "class", "try", "except", "finally", "with")
                stripped = target.lstrip()
                if any(stripped.startswith(kw) for kw in keywords):
                    lines[lineno - 1] = target + ":\n"
                    return "".join(lines)

        # 4. Remove common leading non-code garbage before the first def/class/import.
        return self._trim_preamble(cleaned)

    def _describe_fix(self, exc: SyntaxError) -> str:
        return f"addressed syntax error at line {exc.lineno}: {exc.msg}"

    def _trim_preamble(self, code: str) -> str:
        first = re.search(r"^(def |class |import |from )", code, re.MULTILINE)
        if first:
            start = first.start()
            return code[start:]
        return code

    @staticmethod
    def extract_function(code: str, name: str) -> Any:
        """Return the AST node for ``name`` if it exists."""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        return None
