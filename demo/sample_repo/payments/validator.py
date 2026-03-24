"""Payment validation module for the demo fintech application.

Validates payment data before processing. PCI-sensitive — changes
here require tier 3 review.
"""

import re
from decimal import Decimal


def validate_amount(amount) -> Decimal:
    """Validate and normalize a payment amount.

    Args:
        amount: The raw amount value (may be string, int, float, or Decimal).

    Returns:
        The validated amount as a Decimal.

    Raises:
        ValueError: If the amount is invalid.
    """
    if amount is None:
        raise ValueError("Amount cannot be None")

    try:
        decimal_amount = Decimal(str(amount))
    except Exception as exc:
        raise ValueError(f"Invalid amount format: {amount}") from exc

    if decimal_amount <= 0:
        raise ValueError("Amount must be positive")

    if decimal_amount != decimal_amount.quantize(Decimal("0.01")):
        raise ValueError("Amount cannot have more than 2 decimal places")

    return decimal_amount


def validate_currency(currency: str) -> str:
    """Validate an ISO 4217 currency code.

    Args:
        currency: The currency code to validate.

    Returns:
        The uppercased currency code.

    Raises:
        ValueError: If the currency code is invalid.
    """
    if not currency or not isinstance(currency, str):
        raise ValueError("Currency code is required")

    currency = currency.upper()
    valid_currencies = {"USD", "EUR", "GBP", "CAD", "AUD", "JPY"}

    if currency not in valid_currencies:
        raise ValueError(f"Unsupported currency: {currency}")

    return currency


def validate_card_number(card_number: str) -> bool:
    """Validate a credit card number using Luhn algorithm.

    Args:
        card_number: The card number string (digits only).

    Returns:
        True if the card number passes Luhn validation.

    Raises:
        ValueError: If the card number format is invalid.
    """
    cleaned = re.sub(r"[\s-]", "", card_number)

    if not cleaned.isdigit() or len(cleaned) < 13 or len(cleaned) > 19:
        raise ValueError("Invalid card number format")

    # Luhn algorithm.
    total = 0
    reverse_digits = cleaned[::-1]
    for i, digit in enumerate(reverse_digits):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0
