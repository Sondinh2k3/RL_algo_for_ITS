"""Safe formula evaluator — Python expression hạn chế cho feature derivation.

Cho phép:
  - Số literal (int, float)
  - Biến trong whitelist (ALLOWED_VARS)
  - Toán tử: + - * / ** % // và unary +/-
  - Hàm: min, max, abs, clip(x, lo, hi)
  - So sánh: == != < > <= >= và ternary `x if cond else y`
  - Boolean: and, or

Cấm: import, attribute access, subscript, lambda, comprehension, gọi hàm ngoài
whitelist. Static check ở `validate_formula_syntax` bắt sớm, runtime `_eval`
defense-in-depth (raise FormulaError nếu vẫn lọt).
"""

from __future__ import annotations

import ast
import operator as op
from typing import Any, Mapping


class FormulaError(ValueError):
    """Formula syntax sai hoặc tham chiếu biến/hàm không hợp lệ."""


_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}
_CMP_OPS = {
    ast.Eq: op.eq, ast.NotEq: op.ne,
    ast.Lt: op.lt, ast.LtE: op.le,
    ast.Gt: op.gt, ast.GtE: op.ge,
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


_ALLOWED_FUNCS = {
    "min": min, "max": max, "abs": abs, "clip": _clip,
}


# Tên biến cho phép trong formula. Bao gồm cả realtime (từ Road) và static (từ
# road properties trong bundle). Sim trainer + runtime phải feed cùng tập biến
# này để công thức có ý nghĩa nhất quán hai phía.
ALLOWED_VARS: frozenset[str] = frozenset({
    # Runtime per timestep
    "occupancy",        # % chiếm dụng [0, 100]
    "speed",            # km/h
    "density",          # normalized [0, 1] (nếu cung cấp)
    "queue",            # normalized [0, 1] (nếu cung cấp)
    # Static per road (từ bundle hoặc detector config)
    "lanes",            # số làn
    "length",           # độ dài đường (mét)
    "speed_design",     # tốc độ thiết kế (km/h)
    "saturation_flow",  # vehicles/hour
})


def _eval(node: ast.AST, vars_: Mapping[str, float]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, vars_)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise FormulaError(f"Constant không hỗ trợ: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in vars_:
            return vars_[node.id]
        if node.id in _ALLOWED_FUNCS:
            return _ALLOWED_FUNCS[node.id]
        raise FormulaError(f"Biến/hàm không cho phép: {node.id!r}")
    if isinstance(node, ast.BinOp):
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise FormulaError(f"Toán tử không hỗ trợ: {type(node.op).__name__}")
        return fn(_eval(node.left, vars_), _eval(node.right, vars_))
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise FormulaError(f"Unary op không hỗ trợ: {type(node.op).__name__}")
        return fn(_eval(node.operand, vars_))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, vars_)
        for op_node, right_node in zip(node.ops, node.comparators):
            fn = _CMP_OPS.get(type(op_node))
            if fn is None:
                raise FormulaError(f"So sánh không hỗ trợ: {type(op_node).__name__}")
            right = _eval(right_node, vars_)
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval(v, vars_) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval(v, vars_) for v in node.values)
        raise FormulaError(f"BoolOp không hỗ trợ: {type(node.op).__name__}")
    if isinstance(node, ast.IfExp):
        return _eval(node.body, vars_) if _eval(node.test, vars_) else _eval(node.orelse, vars_)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise FormulaError("Chỉ cho phép gọi min/max/abs/clip.")
        if node.keywords:
            raise FormulaError("Không hỗ trợ keyword arguments.")
        args = [_eval(a, vars_) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise FormulaError(f"Node AST không hỗ trợ: {type(node).__name__}")


def compile_formula(expr: str) -> ast.Expression:
    """Parse + static check AST. Raise FormulaError nếu có node không cho phép."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise FormulaError(f"Syntax không hợp lệ: {e.msg}") from e
    for node in ast.walk(tree):
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda,
                              ast.ListComp, ast.SetComp, ast.DictComp,
                              ast.GeneratorExp, ast.Yield, ast.YieldFrom,
                              ast.Await, ast.Starred)):
            raise FormulaError(
                f"Node {type(node).__name__} bị cấm trong formula."
            )
    return tree


def eval_formula(compiled: ast.Expression, vars_: Mapping[str, float]) -> float:
    """Eval compiled AST với map biến → giá trị. Trả về float."""
    return float(_eval(compiled, vars_))


def validate_formula_syntax(expr: str, allowed_vars: set[str] | frozenset[str] = ALLOWED_VARS) -> None:
    """Static check formula chỉ tham chiếu biến trong allowed_vars."""
    tree = compile_formula(expr)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_FUNCS:
                continue
            if node.id not in allowed_vars:
                raise FormulaError(
                    f"Biến {node.id!r} không có trong allowed_vars={sorted(allowed_vars)}."
                )
