"""Payment processing module for the demo fintech application.

Handles payment creation, validation, and settlement. This module
is PCI-sensitive — changes here require tier 3 (human) review.
"""

from decimal import Decimal
from enum import Enum


class PaymentStatus(str, Enum):
    """Payment lifecycle statuses."""

    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    SETTLED = "settled"
    FAILED = "failed"
    REFUNDED = "refunded"


class PaymentProcessor:
    """Processes payment transactions.

    Attributes:
        gateway: The payment gateway identifier.
        max_amount: Maximum allowed transaction amount.
    """

    def __init__(
        self, gateway: str = "stripe", max_amount: Decimal = Decimal("10000.00")
    ) -> None:
        """Initialize the payment processor.

        Args:
            gateway: Payment gateway to use.
            max_amount: Maximum transaction amount.
        """
        self.gateway = gateway
        self.max_amount = max_amount
        self._transactions: list[dict] = []

    def create_payment(
        self,
        amount: Decimal,
        currency: str = "USD",
        customer_id: str = "",
    ) -> dict:
        """Create a new payment transaction.

        Args:
            amount: The payment amount.
            currency: ISO 4217 currency code.
            customer_id: The customer identifier.

        Returns:
            A dictionary representing the payment record.

        Raises:
            ValueError: If amount is invalid.
        """
        if amount <= 0:
            raise ValueError("Payment amount must be positive")

        if amount > self.max_amount:
            raise ValueError(
                f"Amount {amount} exceeds maximum {self.max_amount}"
            )

        payment = {
            "id": f"pay_{len(self._transactions) + 1:04d}",
            "amount": amount,
            "currency": currency,
            "customer_id": customer_id,
            "status": PaymentStatus.PENDING,
            "gateway": self.gateway,
        }

        self._transactions.append(payment)
        return payment

    def capture_payment(self, payment_id: str) -> dict:
        """Capture an authorized payment.

        Args:
            payment_id: The payment identifier to capture.

        Returns:
            The updated payment record.

        Raises:
            ValueError: If payment not found or not in authorized state.
        """
        payment = self._find_payment(payment_id)
        if payment["status"] != PaymentStatus.AUTHORIZED:
            raise ValueError(
                f"Cannot capture payment in {payment['status']} state"
            )
        payment["status"] = PaymentStatus.CAPTURED
        return payment

    def _find_payment(self, payment_id: str) -> dict:
        """Find a payment by ID.

        Args:
            payment_id: The payment identifier.

        Returns:
            The payment record.

        Raises:
            ValueError: If payment not found.
        """
        for txn in self._transactions:
            if txn["id"] == payment_id:
                return txn
        raise ValueError(f"Payment {payment_id} not found")
