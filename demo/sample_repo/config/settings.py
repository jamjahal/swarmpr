"""Application settings for the demo fintech application.

Configuration tier — changes here get tier 1 (auto-approve eligible).
"""

# Application settings.
APP_NAME = "FinTech Demo"
APP_VERSION = "1.0.0"
DEBUG = False

# Payment settings.
DEFAULT_CURRENCY = "USD"
MAX_PAYMENT_AMOUNT = 10000.00
MIN_PAYMENT_AMOUNT = 0.01

# Auth settings.
TOKEN_TTL_SECONDS = 3600
MAX_LOGIN_ATTEMPTS = 5

# Logging.
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
