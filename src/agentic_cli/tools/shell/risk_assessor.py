"""Risk assessor combining all security layers.

Layer 5: Operation Risk Assessment
- Combines results from tokenizer, classifier, and path analyzer
- Integrates encoding/obfuscation detection
- Computes overall risk level
- Determines approval requirements
"""

from typing import TYPE_CHECKING

from agentic_cli.tools.shell.models import (
    ApprovalType,
    ClassificationResult,
    CommandCategory,
    PathAnalysisResult,
    RiskAssessment,
    RiskLevel,
    TokenizeResult,
)

if TYPE_CHECKING:
    from agentic_cli.tools.shell.config import ShellSecurityConfig
    from agentic_cli.tools.shell.preprocessor import PreprocessResult


class RiskAssessor:
    """Assesses overall risk and determines approval requirements.

    Combines analysis from all layers to compute a final risk score
    and determine what type of approval (if any) is needed.
    """

    def __init__(self, config: "ShellSecurityConfig | None" = None):
        """Initialize risk assessor.

        Args:
            config: Optional ShellSecurityConfig for approval settings.
        """
        self.config = config

    def assess(
        self,
        tokenize_result: TokenizeResult,
        classifications: list[ClassificationResult],
        path_analysis: PathAnalysisResult,
        preprocess_result: "PreprocessResult | None" = None,
    ) -> RiskAssessment:
        """Compute overall risk assessment.

        Args:
            tokenize_result: Result from command tokenization.
            classifications: Results from command classification.
            path_analysis: Result from path analysis.
            preprocess_result: Result from encoding/obfuscation detection.

        Returns:
            RiskAssessment with overall risk and approval requirements.
        """
        assessment = RiskAssessment(
            base_risk=RiskLevel.LOW,
            path_risk=RiskLevel.LOW,
            chaining_risk=RiskLevel.LOW,
            encoding_risk=RiskLevel.LOW,
            tokenize_result=tokenize_result,
            path_analysis=path_analysis,
            preprocess_result=preprocess_result,
        )

        # Store first classification for reference
        if classifications:
            assessment.classification = classifications[0]

        # Assess base risk from command categories
        assessment.base_risk = self._assess_command_risk(classifications)
        if assessment.base_risk == RiskLevel.CRITICAL:
            assessment.block_reasons.extend(
                self._get_block_reasons(classifications)
            )

        # Assess path risk
        assessment.path_risk = self._assess_path_risk(path_analysis)
        assessment.risk_factors.extend(
            self._get_path_risk_factors(path_analysis)
        )

        # Assess chaining complexity risk
        assessment.chaining_risk = self._assess_chaining_risk(tokenize_result)
        if assessment.chaining_risk in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
            assessment.risk_factors.append(
                self._get_chaining_description(tokenize_result)
            )

        # Assess encoding/obfuscation risk
        assessment.encoding_risk = self._assess_encoding_risk(preprocess_result)
        if preprocess_result:
            assessment.risk_factors.extend(
                self._get_encoding_risk_factors(preprocess_result)
            )
            if preprocess_result.block_reason:
                assessment.block_reasons.append(preprocess_result.block_reason)

        # Compute overall risk
        assessment.overall_risk = self._compute_overall_risk(
            assessment.base_risk,
            assessment.path_risk,
            assessment.chaining_risk,
            assessment.encoding_risk,
        )

        # Determine approval requirements
        assessment.approval_required = self._determine_approval(
            assessment, classifications, path_analysis
        )

        return assessment

    def _assess_command_risk(
        self, classifications: list[ClassificationResult]
    ) -> RiskLevel:
        """Assess risk based on command categories."""
        if not classifications:
            return RiskLevel.LOW

        categories = [c.category for c in classifications]

        # BLOCKED -> CRITICAL
        if CommandCategory.BLOCKED in categories:
            return RiskLevel.CRITICAL

        # PRIVILEGED -> HIGH
        if CommandCategory.PRIVILEGED in categories:
            return RiskLevel.HIGH

        # WRITE -> MEDIUM
        if CommandCategory.WRITE in categories:
            return RiskLevel.MEDIUM

        # NETWORK -> MEDIUM
        if CommandCategory.NETWORK in categories:
            return RiskLevel.MEDIUM

        # READ/SAFE -> LOW
        return RiskLevel.LOW

    def _assess_path_risk(self, path_analysis: PathAnalysisResult) -> RiskLevel:
        """Assess risk based on path analysis."""
        # Sensitive paths or traversal -> CRITICAL
        if path_analysis.has_sensitive_paths:
            # Check if it's a write to sensitive path
            if any(p.is_sensitive for p in path_analysis.write_paths):
                return RiskLevel.CRITICAL

        if path_analysis.has_traversal:
            return RiskLevel.HIGH

        # Outside project writes -> HIGH
        if path_analysis.has_outside_project:
            if any(
                not p.in_allowed and not p.in_project
                for p in path_analysis.write_paths
            ):
                return RiskLevel.HIGH

        # Any path outside project -> MEDIUM
        if path_analysis.has_outside_project:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _assess_chaining_risk(self, tokenize_result: TokenizeResult) -> RiskLevel:
        """Assess risk based on command complexity."""
        risk_factors = 0

        if tokenize_result.has_subshells:
            risk_factors += 2

        if tokenize_result.has_chains:
            risk_factors += 1

        if tokenize_result.has_pipes:
            # Pipes are common and often safe
            pass

        if tokenize_result.has_background:
            risk_factors += 1

        # Count total commands
        total_commands = len(tokenize_result.nodes)
        for node in tokenize_result.nodes:
            if node.pipes_to:
                total_commands += 1
            total_commands += len(node.chained_with)

        if total_commands >= 4:
            risk_factors += 2
        elif total_commands >= 2:
            risk_factors += 1

        if risk_factors >= 3:
            return RiskLevel.HIGH
        if risk_factors >= 1:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _assess_encoding_risk(
        self, preprocess_result: "PreprocessResult | None"
    ) -> RiskLevel:
        """Assess risk based on encoding/obfuscation detection."""
        if preprocess_result is None:
            return RiskLevel.LOW

        # If blocked due to decoded dangerous content
        if preprocess_result.block_reason:
            return RiskLevel.CRITICAL

        score = preprocess_result.obfuscation_score

        # High obfuscation score
        if score >= 0.7:
            return RiskLevel.CRITICAL

        if score >= 0.5:
            return RiskLevel.HIGH

        if score >= 0.3:
            return RiskLevel.MEDIUM

        # Any encoding detected warrants at least a warning
        if preprocess_result.has_encoding:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _compute_overall_risk(
        self,
        base_risk: RiskLevel,
        path_risk: RiskLevel,
        chaining_risk: RiskLevel,
        encoding_risk: RiskLevel = RiskLevel.LOW,
    ) -> RiskLevel:
        """Compute overall risk from component risks."""
        risks = [base_risk, path_risk, chaining_risk, encoding_risk]

        # CRITICAL in any -> CRITICAL
        if RiskLevel.CRITICAL in risks:
            return RiskLevel.CRITICAL

        # Two or more HIGH -> CRITICAL
        high_count = risks.count(RiskLevel.HIGH)
        if high_count >= 2:
            return RiskLevel.CRITICAL

        # One HIGH -> HIGH
        if RiskLevel.HIGH in risks:
            return RiskLevel.HIGH

        # Two or more MEDIUM -> HIGH
        medium_count = risks.count(RiskLevel.MEDIUM)
        if medium_count >= 2:
            return RiskLevel.HIGH

        # One MEDIUM -> MEDIUM
        if RiskLevel.MEDIUM in risks:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _determine_approval(
        self,
        assessment: RiskAssessment,
        classifications: list[ClassificationResult],
        path_analysis: PathAnalysisResult,
    ) -> ApprovalType:
        """Determine what approval is required."""
        # CRITICAL -> BLOCK
        if assessment.overall_risk == RiskLevel.CRITICAL:
            return ApprovalType.BLOCK

        # Check config settings
        config = self.config

        # HIGH -> ALWAYS_PROMPT
        if assessment.overall_risk == RiskLevel.HIGH:
            return ApprovalType.ALWAYS_PROMPT

        # MEDIUM risk handling
        if assessment.overall_risk == RiskLevel.MEDIUM:
            # Check if auto-approve is enabled for in-project operations
            if config and config.auto_approve_in_project:
                # Check if all paths are in project
                all_in_project = all(
                    p.in_project or p.in_allowed for p in path_analysis.paths
                )
                if all_in_project:
                    return ApprovalType.AUTO

            return ApprovalType.PROMPT_ONCE

        # LOW risk
        if assessment.overall_risk == RiskLevel.LOW:
            # Check if this is a read-only operation
            is_read_only = all(
                c.category in (CommandCategory.SAFE, CommandCategory.READ)
                for c in classifications
            )

            if config and config.auto_approve_read_only and is_read_only:
                return ApprovalType.AUTO

            # Even LOW risk writes should prompt in strict mode
            if not is_read_only:
                if config and not config.auto_approve_in_project:
                    return ApprovalType.PROMPT_ONCE

            return ApprovalType.AUTO

        return ApprovalType.PROMPT_ONCE

    def _get_block_reasons(
        self, classifications: list[ClassificationResult]
    ) -> list[str]:
        """Get reasons why commands are blocked."""
        reasons = []
        for c in classifications:
            if c.category == CommandCategory.BLOCKED and c.reason:
                reasons.append(c.reason)
        return reasons

    def _get_path_risk_factors(
        self, path_analysis: PathAnalysisResult
    ) -> list[str]:
        """Get path-related risk factors."""
        factors = []

        if path_analysis.has_traversal:
            factors.append("Path traversal detected (..)")

        if path_analysis.has_sensitive_paths:
            sensitive = [p.original for p in path_analysis.paths if p.is_sensitive]
            if sensitive:
                factors.append(f"Sensitive paths: {', '.join(sensitive[:3])}")

        if path_analysis.has_outside_project:
            outside = [
                p.original
                for p in path_analysis.paths
                if not p.in_project and not p.in_allowed
            ]
            if outside:
                factors.append(f"Paths outside project: {', '.join(outside[:3])}")

        return factors

    def _get_chaining_description(self, tokenize_result: TokenizeResult) -> str:
        """Get description of command chaining."""
        parts = []
        if tokenize_result.has_pipes:
            parts.append("pipes")
        if tokenize_result.has_chains:
            parts.append("chains")
        if tokenize_result.has_subshells:
            parts.append("subshells")
        if tokenize_result.has_background:
            parts.append("background execution")

        return f"Complex command with {', '.join(parts)}"

    def _get_encoding_risk_factors(
        self, preprocess_result: "PreprocessResult"
    ) -> list[str]:
        """Get encoding-related risk factors."""
        factors = []

        if preprocess_result.encodings_detected:
            factors.append(
                f"Encoding detected: {', '.join(preprocess_result.encodings_detected)}"
            )

        if preprocess_result.obfuscation_score > 0:
            factors.append(
                f"Obfuscation score: {preprocess_result.obfuscation_score:.2f}"
            )

        # Add warnings (but limit to avoid verbose output)
        for warning in preprocess_result.warnings[:3]:
            factors.append(warning)

        return factors
