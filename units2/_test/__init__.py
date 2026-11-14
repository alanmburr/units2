from __future__ import annotations

import math
import unittest
from typing import SupportsFloat, SupportsIndex


class TestCase(unittest.TestCase):
    def assertIsClose(
        self,
        a: SupportsFloat | SupportsIndex,
        b: SupportsFloat | SupportsIndex,
        rel_tol: SupportsFloat | SupportsIndex = 1e-09,
        abs_tol: SupportsFloat | SupportsIndex = 0.0,
        msg: str | None = None
    ):
        isclose = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        a, b = float(a), float(b)
        rel_tol, abs_tol = float(rel_tol), float(abs_tol)
        if isclose:
            return
        raise self.failureException(
            f"(|{a} - {b}| = {abs(a - b)})"
            " ≤ "
            f"({max(rel_tol * max(abs(a), abs(b)), abs_tol)} = "
            "(max( rel_tol * max(|a|, |b|), abs_tol))" +
            (f" : {msg}" if msg is not None else "")
        )
