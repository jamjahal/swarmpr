"""Tests for the payment processor."""

from decimal import Decimal

import pytest

from payments.processor import PaymentProcessor, PaymentStatus


class TestPaymentProcessor:
    """Tests for PaymentProcessor."""

    def test_create_payment(self):
        """Test creating a valid payment."""
        processor = PaymentProcessor()
        payment = processor.create_payment(
            amount=Decimal("25.00"),
            currency="USD",
            customer_id="cust_001",
        )
        assert payment["status"] == PaymentStatus.PENDING
        assert payment["amount"] == Decimal("25.00")

    def test_create_payment_negative_amount(self):
        """Test that negative amounts are rejected."""
        processor = PaymentProcessor()
        with pytest.raises(ValueError, match="positive"):
            processor.create_payment(amount=Decimal("-10.00"))

    def test_create_payment_exceeds_max(self):
        """Test that amounts over the max are rejected."""
        processor = PaymentProcessor(max_amount=Decimal("100.00"))
        with pytest.raises(ValueError, match="exceeds"):
            processor.create_payment(amount=Decimal("200.00"))
