"""Authentication token management for the demo fintech application.

Handles token generation, validation, and revocation. Auth-sensitive —
changes here require tier 3 review.
"""

import hashlib
import secrets
import time


class TokenManager:
    """Manages authentication tokens.

    Attributes:
        token_ttl: Token time-to-live in seconds.
    """

    def __init__(self, token_ttl: int = 3600) -> None:
        """Initialize the token manager.

        Args:
            token_ttl: Token expiration time in seconds.
        """
        self.token_ttl = token_ttl
        self._tokens: dict[str, dict] = {}

    def generate_token(self, user_id: str) -> str:
        """Generate a new authentication token for a user.

        Args:
            user_id: The user identifier.

        Returns:
            The generated token string.
        """
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        self._tokens[token_hash] = {
            "user_id": user_id,
            "created_at": time.time(),
            "expires_at": time.time() + self.token_ttl,
        }

        return raw_token

    def validate_token(self, token: str) -> dict | None:
        """Validate a token and return its metadata.

        Args:
            token: The raw token string to validate.

        Returns:
            Token metadata dict if valid, None if invalid or expired.
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        metadata = self._tokens.get(token_hash)

        if not metadata:
            return None

        if time.time() > metadata["expires_at"]:
            del self._tokens[token_hash]
            return None

        return metadata

    def revoke_token(self, token: str) -> bool:
        """Revoke an authentication token.

        Args:
            token: The raw token string to revoke.

        Returns:
            True if the token was found and revoked, False otherwise.
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if token_hash in self._tokens:
            del self._tokens[token_hash]
            return True
        return False
