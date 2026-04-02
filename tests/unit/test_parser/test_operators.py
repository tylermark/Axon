"""Unit tests for src/parser/operators.py — PostScript operator registry."""

from __future__ import annotations

import pytest

from src.parser.operators import (
    OPERATOR_REGISTRY,
    OperatorCategory,
    OperatorInfo,
    OperatorType,
    get_operator_info,
    is_paint_operator,
    is_path_operator,
    is_state_operator,
)

# ---------------------------------------------------------------------------
# OperatorType enum
# ---------------------------------------------------------------------------


class TestOperatorType:
    def test_enum_values(self) -> None:
        assert OperatorType.MOVETO == "moveto"
        assert OperatorType.LINETO == "lineto"
        assert OperatorType.CURVETO == "curveto"
        assert OperatorType.CLOSEPATH == "closepath"

    def test_membership(self) -> None:
        assert "moveto" in OperatorType.__members__.values()
        assert "lineto" in OperatorType.__members__.values()
        assert "curveto" in OperatorType.__members__.values()
        assert "closepath" in OperatorType.__members__.values()

    def test_enum_count(self) -> None:
        assert len(OperatorType) == 4


# ---------------------------------------------------------------------------
# OperatorCategory enum
# ---------------------------------------------------------------------------


class TestOperatorCategory:
    def test_enum_values(self) -> None:
        assert OperatorCategory.PATH_CONSTRUCTION == "path_construction"
        assert OperatorCategory.PATH_PAINTING == "path_painting"
        assert OperatorCategory.GRAPHICS_STATE == "graphics_state"
        assert OperatorCategory.COLOR == "color"
        assert OperatorCategory.CLIPPING == "clipping"

    def test_enum_count(self) -> None:
        assert len(OperatorCategory) == 5


# ---------------------------------------------------------------------------
# OPERATOR_REGISTRY
# ---------------------------------------------------------------------------


class TestOperatorRegistry:
    def test_registry_has_expected_operators(self) -> None:
        expected = ["m", "l", "c", "h", "q", "Q", "cm", "S", "W"]
        for op in expected:
            assert op in OPERATOR_REGISTRY, f"Missing operator '{op}'"

    def test_registry_operator_count(self) -> None:
        # Registry should have ~39 operators
        assert len(OPERATOR_REGISTRY) >= 35
        assert len(OPERATOR_REGISTRY) <= 45

    def test_registry_values_are_operator_info(self) -> None:
        for op, info in OPERATOR_REGISTRY.items():
            assert isinstance(info, OperatorInfo), f"'{op}' is not OperatorInfo"

    def test_operator_pdf_operator_matches_key(self) -> None:
        for op, info in OPERATOR_REGISTRY.items():
            assert info.pdf_operator == op


# ---------------------------------------------------------------------------
# is_path_operator
# ---------------------------------------------------------------------------


class TestIsPathOperator:
    @pytest.mark.parametrize("op", ["m", "l", "c", "v", "y", "h", "re"])
    def test_true_for_path_construction(self, op: str) -> None:
        assert is_path_operator(op) is True

    @pytest.mark.parametrize("op", ["S", "q", "Q", "f", "n"])
    def test_false_for_non_path(self, op: str) -> None:
        assert is_path_operator(op) is False

    def test_false_for_unknown(self) -> None:
        assert is_path_operator("UNKNOWN_OP") is False


# ---------------------------------------------------------------------------
# is_paint_operator
# ---------------------------------------------------------------------------


class TestIsPaintOperator:
    @pytest.mark.parametrize("op", ["S", "s", "f", "B", "n", "F", "f*", "B*", "b", "b*"])
    def test_true_for_paint(self, op: str) -> None:
        assert is_paint_operator(op) is True

    @pytest.mark.parametrize("op", ["m", "l", "q", "cm"])
    def test_false_for_non_paint(self, op: str) -> None:
        assert is_paint_operator(op) is False


# ---------------------------------------------------------------------------
# is_state_operator
# ---------------------------------------------------------------------------


class TestIsStateOperator:
    @pytest.mark.parametrize("op", ["q", "Q", "cm", "w", "G", "RG", "K"])
    def test_true_for_state(self, op: str) -> None:
        assert is_state_operator(op) is True

    @pytest.mark.parametrize("op", ["m", "l", "S", "f"])
    def test_false_for_non_state(self, op: str) -> None:
        assert is_state_operator(op) is False


# ---------------------------------------------------------------------------
# get_operator_info
# ---------------------------------------------------------------------------


class TestGetOperatorInfo:
    def test_returns_info_for_known_ops(self) -> None:
        info = get_operator_info("m")
        assert info is not None
        assert info.name == "moveto"
        assert info.category == OperatorCategory.PATH_CONSTRUCTION

    def test_returns_none_for_unknown(self) -> None:
        assert get_operator_info("NONEXISTENT") is None

    def test_param_count_m(self) -> None:
        info = get_operator_info("m")
        assert info is not None
        assert info.param_count == 2

    def test_param_count_l(self) -> None:
        info = get_operator_info("l")
        assert info is not None
        assert info.param_count == 2

    def test_param_count_c(self) -> None:
        info = get_operator_info("c")
        assert info is not None
        assert info.param_count == 6

    def test_param_count_h(self) -> None:
        info = get_operator_info("h")
        assert info is not None
        assert info.param_count == 0

    def test_param_count_cm(self) -> None:
        info = get_operator_info("cm")
        assert info is not None
        assert info.param_count == 6

    def test_param_count_w(self) -> None:
        info = get_operator_info("w")
        assert info is not None
        assert info.param_count == 1

    def test_param_count_re(self) -> None:
        info = get_operator_info("re")
        assert info is not None
        assert info.param_count == 4

    def test_param_count_sc_is_variable(self) -> None:
        info = get_operator_info("SC")
        assert info is not None
        assert info.param_count is None

    def test_modifies_path_flag(self) -> None:
        m = get_operator_info("m")
        assert m is not None and m.modifies_path is True
        s = get_operator_info("S")
        assert s is not None and s.modifies_path is False

    def test_modifies_state_flag(self) -> None:
        cm = get_operator_info("cm")
        assert cm is not None and cm.modifies_state is True
        m = get_operator_info("m")
        assert m is not None and m.modifies_state is False
