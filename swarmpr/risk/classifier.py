"""Blast radius classification engine for SwarmPR.

Classifies code changes by risk tier based on file paths and content
keywords. This drives the escalation framework that determines whether
AI approval is sufficient or human review is mandatory.
"""

from pydantic import BaseModel, Field

from swarmpr.config import RiskTierConfig

# Tier number to action name mapping (higher tier = higher risk).
_TIER_NUMBERS = {"tier_3": 3, "tier_2": 2, "tier_1": 1}
_DEFAULT_TIER = 2
_DEFAULT_ACTION = "flag"


class ClassificationResult(BaseModel):
    """Result of a blast radius classification.

    Attributes:
        tier: Risk tier (1-3, higher = more risk).
        action: Escalation action (approve/flag/block).
        justification: Human-readable explanation of the classification.
        matched_rules: List of rules that triggered the classification.
    """

    tier: int
    action: str
    justification: str = ""
    matched_rules: list[str] = Field(default_factory=list)


class RiskClassifier:
    """Classifies code changes by blast radius tier.

    Uses a combination of file path prefix matching and content keyword
    detection to assign risk tiers. The highest matching tier wins.
    Conservative by design — when in doubt, escalate up.

    Attributes:
        risk_tiers: The configured risk tier definitions.
    """

    def __init__(self, risk_tiers: dict[str, RiskTierConfig]) -> None:
        """Initialize the classifier with risk tier configurations.

        Args:
            risk_tiers: Mapping of tier names to their configurations.
        """
        self.risk_tiers = risk_tiers

    def classify_paths(self, paths: list[str]) -> ClassificationResult:
        """Classify a set of file paths by their highest risk tier.

        Args:
            paths: List of relative file paths to classify.

        Returns:
            A ClassificationResult with the highest matching tier.
        """
        if not paths:
            return ClassificationResult(
                tier=1,
                action="approve",
                justification="No files changed.",
                matched_rules=[],
            )

        highest_tier = 0
        highest_action = "approve"
        matched_rules: list[str] = []

        for path in paths:
            tier_num, action, rule = self._match_path(path)
            if rule:
                matched_rules.append(rule)
            if tier_num > highest_tier:
                highest_tier = tier_num
                highest_action = action

        if highest_tier == 0:
            return ClassificationResult(
                tier=_DEFAULT_TIER,
                action=_DEFAULT_ACTION,
                justification=f"No tier matched for paths: {paths}. "
                f"Defaulting to tier {_DEFAULT_TIER}.",
                matched_rules=matched_rules,
            )

        justification_parts = [
            f"Matched tier {highest_tier} ({highest_action})"
        ]
        if matched_rules:
            justification_parts.append(
                f"triggered by: {', '.join(matched_rules)}"
            )

        return ClassificationResult(
            tier=highest_tier,
            action=highest_action,
            justification=" — ".join(justification_parts),
            matched_rules=matched_rules,
        )

    def classify_content(
        self, path: str, content: str
    ) -> ClassificationResult:
        """Classify a file by its path and content keywords.

        First classifies by path, then checks content for keywords
        that might escalate the tier.

        Args:
            path: Relative file path.
            content: The file content or diff text.

        Returns:
            A ClassificationResult reflecting both path and content.
        """
        path_result = self.classify_paths([path])

        content_lower = content.lower()
        keyword_tier = 0
        keyword_action = "approve"
        keyword_rules: list[str] = []

        for tier_name, tier_config in self.risk_tiers.items():
            tier_num = _TIER_NUMBERS.get(tier_name, _DEFAULT_TIER)
            for keyword in tier_config.keywords:
                if keyword.lower() in content_lower:
                    keyword_rules.append(
                        f"keyword: '{keyword}' in {path}"
                    )
                    if tier_num > keyword_tier:
                        keyword_tier = tier_num
                        keyword_action = tier_config.action.value

        if keyword_tier > path_result.tier:
            all_rules = path_result.matched_rules + keyword_rules
            return ClassificationResult(
                tier=keyword_tier,
                action=keyword_action,
                justification=(
                    f"Keyword match escalated from tier "
                    f"{path_result.tier} to tier {keyword_tier}"
                ),
                matched_rules=all_rules,
            )

        return path_result

    def classify_changes(
        self, changes: list[dict[str, str]]
    ) -> ClassificationResult:
        """Classify a set of file changes by combined path and content.

        This is the main entry point used by the planner and reviewer
        agents. It evaluates all changes and returns the highest tier.

        Args:
            changes: List of dicts with 'path' and optional 'content' keys.

        Returns:
            A ClassificationResult reflecting the highest risk across
            all changes.
        """
        if not changes:
            return ClassificationResult(
                tier=1,
                action="approve",
                justification="No changes to classify.",
                matched_rules=[],
            )

        highest_result = ClassificationResult(
            tier=0, action="approve"
        )
        all_rules: list[str] = []

        for change in changes:
            path = change.get("path", "")
            content = change.get("content", "")

            if content:
                result = self.classify_content(path, content)
            else:
                result = self.classify_paths([path])

            all_rules.extend(result.matched_rules)

            if result.tier > highest_result.tier:
                highest_result = result

        highest_result.matched_rules = all_rules

        if not highest_result.justification:
            paths = [c.get("path", "") for c in changes]
            highest_result.justification = (
                f"Classified {len(changes)} file(s): {', '.join(paths)}"
            )

        return highest_result

    def _match_path(self, path: str) -> tuple[int, str, str]:
        """Match a single file path against configured tier prefixes.

        Args:
            path: Relative file path to match.

        Returns:
            A tuple of (tier_number, action_name, matched_rule_string).
            Returns (0, 'approve', '') if no tier matches.
        """
        for tier_name, tier_config in self.risk_tiers.items():
            tier_num = _TIER_NUMBERS.get(tier_name, _DEFAULT_TIER)
            for prefix in tier_config.paths:
                if path.startswith(prefix):
                    return (
                        tier_num,
                        tier_config.action.value,
                        f"path: {prefix}",
                    )
        return (0, "approve", "")
