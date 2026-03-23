"""Escalation policy engine for SwarmPR.

Maps risk classification results to concrete escalation decisions,
factoring in diff size limits, forbidden paths, and tier-based rules.
This is the "should a human review this?" decision maker.
"""

from pydantic import BaseModel

from swarmpr.config import PipelineConfig, RiskTierConfig
from swarmpr.risk.classifier import ClassificationResult


class EscalationDecision(BaseModel):
    """The final escalation decision for a pipeline run.

    Attributes:
        action: The escalation action (approve/flag/block).
        tier: The risk tier that drove the decision.
        can_auto_merge: Whether the PR can be merged without human input.
        requires_human_review: Whether a human must review before merge.
        reason: Human-readable explanation of the decision.
    """

    action: str
    tier: int
    can_auto_merge: bool
    requires_human_review: bool
    reason: str


class EscalationPolicy:
    """Evaluates classification results against policy rules.

    Combines tier-based rules with pipeline constraints (diff size,
    forbidden paths) to produce a final escalation decision.

    Attributes:
        risk_tiers: The configured risk tier definitions.
        pipeline_config: Pipeline execution constraints.
    """

    def __init__(
        self,
        risk_tiers: dict[str, RiskTierConfig],
        pipeline_config: PipelineConfig,
    ) -> None:
        """Initialize the policy engine.

        Args:
            risk_tiers: Mapping of tier names to their configurations.
            pipeline_config: Pipeline constraints (diff limits, etc.).
        """
        self.risk_tiers = risk_tiers
        self.pipeline_config = pipeline_config

    def evaluate(
        self,
        classification: ClassificationResult,
        diff_lines: int = 0,
        changed_paths: list[str] | None = None,
    ) -> EscalationDecision:
        """Evaluate a classification result and return an escalation decision.

        Checks tier-based rules first, then applies overrides for
        diff size limits and forbidden path violations.

        Args:
            classification: The risk classification result.
            diff_lines: Number of lines in the diff.
            changed_paths: List of changed file paths to check
                against forbidden paths.

        Returns:
            An EscalationDecision with the final verdict.
        """
        changed_paths = changed_paths or []

        # Check forbidden paths first — always blocks.
        forbidden_hit = self._check_forbidden_paths(changed_paths)
        if forbidden_hit:
            return EscalationDecision(
                action="block",
                tier=classification.tier,
                can_auto_merge=False,
                requires_human_review=True,
                reason=(
                    f"Forbidden path(s) modified: {forbidden_hit}. "
                    f"Human review required."
                ),
            )

        # Check diff size — escalates if over limit.
        if diff_lines > self.pipeline_config.max_diff_lines:
            return EscalationDecision(
                action="block",
                tier=classification.tier,
                can_auto_merge=False,
                requires_human_review=True,
                reason=(
                    f"Diff size ({diff_lines} lines) exceeds maximum "
                    f"({self.pipeline_config.max_diff_lines}). "
                    f"Human review required."
                ),
            )

        # Apply tier-based rules.
        return self._tier_decision(classification)

    def _tier_decision(
        self, classification: ClassificationResult
    ) -> EscalationDecision:
        """Map a classification tier to an escalation decision.

        Args:
            classification: The risk classification result.

        Returns:
            An EscalationDecision based on the tier action.
        """
        action = classification.action

        if action == "approve":
            return EscalationDecision(
                action="approve",
                tier=classification.tier,
                can_auto_merge=True,
                requires_human_review=False,
                reason=(
                    f"Tier {classification.tier} — "
                    f"{classification.justification}. "
                    f"Auto-approval eligible."
                ),
            )
        elif action == "flag":
            return EscalationDecision(
                action="flag",
                tier=classification.tier,
                can_auto_merge=False,
                requires_human_review=False,
                reason=(
                    f"Tier {classification.tier} — "
                    f"{classification.justification}. "
                    f"AI-reviewed, human may review."
                ),
            )
        else:  # block
            return EscalationDecision(
                action="block",
                tier=classification.tier,
                can_auto_merge=False,
                requires_human_review=True,
                reason=(
                    f"Tier {classification.tier} — "
                    f"{classification.justification}. "
                    f"Human review mandatory."
                ),
            )

    def _check_forbidden_paths(
        self, changed_paths: list[str]
    ) -> list[str]:
        """Check if any changed paths match forbidden path patterns.

        Args:
            changed_paths: List of changed file paths.

        Returns:
            List of forbidden paths that were matched, empty if none.
        """
        violations = []
        for path in changed_paths:
            for forbidden in self.pipeline_config.forbidden_paths:
                if path == forbidden or path.startswith(forbidden):
                    violations.append(path)
                    break
        return violations
