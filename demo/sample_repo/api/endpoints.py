"""REST API endpoints for the demo fintech application.

Business logic tier — changes here get tier 2 (AI review sufficient).
"""

from decimal import Decimal


class PaymentAPI:
    """API handler for payment operations.

    Attributes:
        processor: The payment processor instance.
    """

    def __init__(self, processor) -> None:
        """Initialize with a payment processor.

        Args:
            processor: A PaymentProcessor instance.
        """
        self.processor = processor

    def create_payment_endpoint(self, request: dict) -> dict:
        """Handle POST /payments.

        Args:
            request: The request body with amount, currency, customer_id.

        Returns:
            A response dict with the payment record or error.
        """
        try:
            amount = Decimal(str(request.get("amount", 0)))
            currency = request.get("currency", "USD")
            customer_id = request.get("customer_id", "")

            payment = self.processor.create_payment(
                amount=amount,
                currency=currency,
                customer_id=customer_id,
            )

            return {
                "status": "success",
                "data": {
                    "id": payment["id"],
                    "amount": str(payment["amount"]),
                    "currency": payment["currency"],
                    "status": payment["status"],
                },
            }

        except ValueError as exc:
            return {"status": "error", "message": str(exc)}

    def get_payment_endpoint(self, payment_id: str) -> dict:
        """Handle GET /payments/{id}.

        Args:
            payment_id: The payment identifier.

        Returns:
            A response dict with the payment record or error.
        """
        try:
            payment = self.processor._find_payment(payment_id)
            return {
                "status": "success",
                "data": {
                    "id": payment["id"],
                    "amount": str(payment["amount"]),
                    "currency": payment["currency"],
                    "status": payment["status"],
                },
            }
        except ValueError as exc:
            return {"status": "error", "message": str(exc)}
