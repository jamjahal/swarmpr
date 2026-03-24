"""Tests for payment validation."""

from decimal import Decimal

import pytest

from payments.validator import validate_amount, validate_currency


class TestValidateAmount:
    """Tests for validate_amount."""

    def test_valid_decimal(self):
        """Test validating a valid decimal amount."""
        result = validate_amount(Decimal("25.00"))
        assert result == Decimal("25.00")

    def test_valid_string(self):
        """Test validating a string amount."""
        result = validate_amount("99.99")
        assert result == Decimal("99.99")

    def test_none_raises(self):
        """Test that None amount raises ValueError."""
        with pytest.raises(ValueError, match="None"):
            validate_amount(None)

    def test_negative_raises(self):
        """Test that negative amount raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            validate_amount("-5.00")


class TestValidateCurrency:
    """Tests for validate_currency."""

    def test_valid_usd(self):
        """Test validating USD."""
        assert validate_currency("USD") == "USD"

    def test_lowercase_normalized(self):
        """Test that lowercase codes are uppercased."""
        assert validate_currency("eur") == "EUR"

    def test_unsupported_currency(self):
        """Test that unsupported currencies raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            validate_currency("XYZ")
