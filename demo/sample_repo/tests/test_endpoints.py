"""Tests for API endpoints."""

from decimal import Decimal

from api.endpoints import PaymentAPI
from payments.processor import PaymentProcessor


class TestPaymentAPI:
    """Tests for PaymentAPI endpoints."""

    def test_create_payment_success(self):
        """Test successful payment creation via API."""
        processor = PaymentProcessor()
        api = PaymentAPI(processor)
        response = api.create_payment_endpoint({
            "amount": "50.00",
            "currency": "USD",
            "customer_id": "cust_001",
        })
        assert response["status"] == "success"

    def test_create_payment_invalid_amount(self):
        """Test payment creation with invalid amount."""
        processor = PaymentProcessor()
        api = PaymentAPI(processor)
        response = api.create_payment_endpoint({
            "amount": "-10.00",
            "currency": "USD",
        })
        assert response["status"] == "error"
