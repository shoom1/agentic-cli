"""Input preprocessor for detecting and neutralizing obfuscation.

Layer 1: Input Preprocessing & Decoding
- Detect encoding/obfuscation techniques
- Compute obfuscation score
- Block or warn based on detected patterns
"""

import base64
import re
from dataclasses import dataclass, field
from urllib.parse import unquote


@dataclass
class PreprocessResult:
    """Result of preprocessing a command for obfuscation detection.

    Attributes:
        original_command: The original command string.
        normalized_command: Command after decoding (if applicable).
        encodings_detected: List of encoding types found.
        obfuscation_score: Score from 0-1 indicating obfuscation level.
        block_reason: Reason for blocking if score > threshold.
        decoded_payloads: Any decoded content found.
        warnings: Warning messages for detected patterns.
    """
    original_command: str
    normalized_command: str
    encodings_detected: list[str] = field(default_factory=list)
    obfuscation_score: float = 0.0
    block_reason: str | None = None
    decoded_payloads: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_blocked(self) -> bool:
        """Check if command should be blocked due to obfuscation."""
        return self.block_reason is not None

    @property
    def has_encoding(self) -> bool:
        """Check if any encoding was detected."""
        return len(self.encodings_detected) > 0


class InputPreprocessor:
    """Detects and analyzes encoding/obfuscation in shell commands.

    Scans for various obfuscation techniques:
    - Base64 encoding
    - Hex encoding
    - Octal encoding
    - URL encoding
    - Unicode homoglyphs
    - Variable expansion tricks
    """

    # Threshold for automatic blocking
    BLOCK_THRESHOLD = 0.7

    # Base64 patterns
    BASE64_COMMAND_PATTERN = re.compile(
        r"base64\s+(-d|--decode)", re.IGNORECASE
    )
    BASE64_ECHO_PATTERN = re.compile(
        r"echo\s+['\"]?([A-Za-z0-9+/=]{8,})['\"]?\s*\|\s*base64\s+(-d|--decode)",
        re.IGNORECASE
    )
    BASE64_INLINE_PATTERN = re.compile(
        r"\$\(echo\s+['\"]?([A-Za-z0-9+/=]{8,})['\"]?\s*\|\s*base64\s+(-d|--decode)\)",
        re.IGNORECASE
    )
    # Standalone base64 strings (long enough to be suspicious)
    BASE64_STRING_PATTERN = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")

    # Hex patterns
    HEX_ESCAPE_PATTERN = re.compile(r"\\x[0-9a-fA-F]{2}")
    XXD_DECODE_PATTERN = re.compile(r"xxd\s+(-r|--reverse)", re.IGNORECASE)
    PRINTF_HEX_PATTERN = re.compile(r"printf\s+['\"]\\x")

    # Octal patterns - detect $'\nnn' style escapes
    # Must match the actual escape sequences, not literal backslash chars
    OCTAL_ESCAPE_PATTERN = re.compile(r"\$'(\\[0-7]{3})+'")
    PRINTF_OCTAL_PATTERN = re.compile(r"printf\s+['\"]\\[0-7]{3}")

    # URL encoding patterns
    URL_ENCODED_PATTERN = re.compile(r"%[0-9a-fA-F]{2}")

    # Unicode patterns
    UNICODE_ESCAPE_PATTERN = re.compile(r"\$'\\u[0-9a-fA-F]{4}'")
    UNICODE_LONG_ESCAPE_PATTERN = re.compile(r"\$'\\U[0-9a-fA-F]{8}'")

    # Common homoglyphs that look like ASCII but aren't
    # These can be used to bypass filters
    HOMOGLYPH_MAP = {
        "\u0430": "a",  # Cyrillic а
        "\u0435": "e",  # Cyrillic е
        "\u043e": "o",  # Cyrillic о
        "\u0440": "p",  # Cyrillic р
        "\u0441": "c",  # Cyrillic с
        "\u0443": "y",  # Cyrillic у
        "\u0445": "x",  # Cyrillic х
        "\u0456": "i",  # Cyrillic і
        "\u0458": "j",  # Cyrillic ј
        "\u04bb": "h",  # Cyrillic һ
        "\u0501": "d",  # Cyrillic ԁ
        "\u051b": "q",  # Cyrillic ԛ
        "\uff52": "r",  # Fullwidth r
        "\uff4d": "m",  # Fullwidth m
        "\u2212": "-",  # Minus sign (not hyphen)
    }

    # Dangerous patterns in decoded content
    DANGEROUS_DECODED_PATTERNS = [
        re.compile(r"rm\s+-[rf]", re.IGNORECASE),
        re.compile(r"/bin/(ba)?sh", re.IGNORECASE),
        re.compile(r"curl.*\|.*sh", re.IGNORECASE),
        re.compile(r"wget.*\|.*sh", re.IGNORECASE),
        re.compile(r"sudo", re.IGNORECASE),
        re.compile(r"chmod\s+[0-7]{3,4}", re.IGNORECASE),
        re.compile(r"/etc/(passwd|shadow)", re.IGNORECASE),
        re.compile(r"nc\s+-[el]", re.IGNORECASE),  # netcat listener
        re.compile(r"python.*-c.*import", re.IGNORECASE),
        re.compile(r"perl\s+-e", re.IGNORECASE),
    ]

    def process(self, command: str) -> PreprocessResult:
        """Process a command to detect encoding and obfuscation.

        Args:
            command: The shell command to analyze.

        Returns:
            PreprocessResult with detection results.
        """
        result = PreprocessResult(
            original_command=command,
            normalized_command=command,
        )

        # Run all detection methods
        score = 0.0

        # Check for base64
        base64_score = self._detect_base64(command, result)
        score += base64_score

        # Check for hex encoding
        hex_score = self._detect_hex(command, result)
        score += hex_score

        # Check for octal encoding
        octal_score = self._detect_octal(command, result)
        score += octal_score

        # Check for URL encoding
        url_score = self._detect_url_encoding(command, result)
        score += url_score

        # Check for unicode/homoglyphs
        unicode_score = self._detect_unicode(command, result)
        score += unicode_score

        # Check for variable expansion tricks
        var_score = self._detect_variable_tricks(command, result)
        score += var_score

        # Cap score at 1.0
        result.obfuscation_score = min(score, 1.0)

        # Check if we should block
        if result.obfuscation_score >= self.BLOCK_THRESHOLD:
            result.block_reason = (
                f"High obfuscation score ({result.obfuscation_score:.2f}): "
                f"detected {', '.join(result.encodings_detected)}"
            )

        # Check decoded payloads for dangerous content
        self._check_decoded_content(result)

        return result

    def _detect_base64(self, command: str, result: PreprocessResult) -> float:
        """Detect base64 encoding patterns."""
        score = 0.0

        # Check for explicit base64 decode commands
        if self.BASE64_COMMAND_PATTERN.search(command):
            result.encodings_detected.append("base64")
            result.warnings.append("Base64 decode command detected")
            score += 0.3

        # Check for echo | base64 -d pattern
        match = self.BASE64_ECHO_PATTERN.search(command)
        if match:
            score += 0.4
            encoded = match.group(1)
            try:
                decoded = base64.b64decode(encoded).decode("utf-8", errors="ignore")
                result.decoded_payloads.append(f"base64: {decoded}")
            except Exception:
                pass

        # Check for inline base64 decode
        match = self.BASE64_INLINE_PATTERN.search(command)
        if match:
            score += 0.5
            encoded = match.group(1)
            try:
                decoded = base64.b64decode(encoded).decode("utf-8", errors="ignore")
                result.decoded_payloads.append(f"base64_inline: {decoded}")
            except Exception:
                pass

        # Check for suspicious base64 strings
        matches = self.BASE64_STRING_PATTERN.findall(command)
        for match in matches:
            if len(match) > 40:  # Long base64 string
                # Try to decode to see if it's valid
                try:
                    decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
                    # Check if decoded looks like a command
                    if any(c in decoded for c in ["/", "|", "&", ";"]):
                        result.encodings_detected.append("base64_string")
                        result.warnings.append(f"Suspicious base64 string: {match[:20]}...")
                        result.decoded_payloads.append(f"base64_potential: {decoded}")
                        score += 0.3
                except Exception:
                    pass

        return score

    def _detect_hex(self, command: str, result: PreprocessResult) -> float:
        """Detect hex encoding patterns."""
        score = 0.0

        # Count hex escapes
        hex_matches = self.HEX_ESCAPE_PATTERN.findall(command)
        if len(hex_matches) >= 3:
            result.encodings_detected.append("hex")
            result.warnings.append(f"Hex escapes detected: {len(hex_matches)} occurrences")
            score += 0.2 + (min(len(hex_matches), 10) * 0.03)

            # Try to decode hex sequence
            try:
                hex_sequence = "".join(hex_matches)
                decoded = bytes.fromhex(
                    hex_sequence.replace("\\x", "")
                ).decode("utf-8", errors="ignore")
                if decoded:
                    result.decoded_payloads.append(f"hex: {decoded}")
            except Exception:
                pass

        # Check for xxd reverse
        if self.XXD_DECODE_PATTERN.search(command):
            result.encodings_detected.append("xxd")
            result.warnings.append("xxd reverse (hex decode) detected")
            score += 0.3

        # Check for printf with hex
        if self.PRINTF_HEX_PATTERN.search(command):
            result.encodings_detected.append("printf_hex")
            result.warnings.append("printf with hex escapes detected")
            score += 0.25

        return score

    def _detect_octal(self, command: str, result: PreprocessResult) -> float:
        """Detect octal encoding patterns."""
        score = 0.0

        # Check for $'\...' octal escapes
        octal_matches = self.OCTAL_ESCAPE_PATTERN.findall(command)
        if octal_matches:
            result.encodings_detected.append("octal")
            result.warnings.append(f"Octal escapes detected: {len(octal_matches)} occurrences")
            score += 0.2 + (min(len(octal_matches), 10) * 0.03)

        # Check for printf with octal
        if self.PRINTF_OCTAL_PATTERN.search(command):
            result.encodings_detected.append("printf_octal")
            result.warnings.append("printf with octal escapes detected")
            score += 0.25

        return score

    def _detect_url_encoding(self, command: str, result: PreprocessResult) -> float:
        """Detect URL encoding patterns."""
        score = 0.0

        url_matches = self.URL_ENCODED_PATTERN.findall(command)
        if len(url_matches) >= 3:
            result.encodings_detected.append("url")
            result.warnings.append(f"URL encoding detected: {len(url_matches)} occurrences")
            score += 0.15 + (min(len(url_matches), 10) * 0.02)

            # Try to decode
            try:
                decoded = unquote(command)
                if decoded != command:
                    result.decoded_payloads.append(f"url: {decoded}")
            except Exception:
                pass

        return score

    def _detect_unicode(self, command: str, result: PreprocessResult) -> float:
        """Detect unicode/homoglyph obfuscation."""
        score = 0.0

        # Check for unicode escapes
        if self.UNICODE_ESCAPE_PATTERN.search(command):
            result.encodings_detected.append("unicode_escape")
            result.warnings.append("Unicode escape sequences detected")
            score += 0.3

        if self.UNICODE_LONG_ESCAPE_PATTERN.search(command):
            result.encodings_detected.append("unicode_long_escape")
            result.warnings.append("Long unicode escape sequences detected")
            score += 0.35

        # Check for homoglyphs
        homoglyphs_found = []
        for char, ascii_equiv in self.HOMOGLYPH_MAP.items():
            if char in command:
                homoglyphs_found.append(f"{char}->{ascii_equiv}")

        if homoglyphs_found:
            result.encodings_detected.append("homoglyphs")
            result.warnings.append(f"Homoglyph characters detected: {', '.join(homoglyphs_found[:5])}")
            score += 0.4 + (min(len(homoglyphs_found), 5) * 0.05)

            # Create normalized version
            normalized = command
            for char, ascii_equiv in self.HOMOGLYPH_MAP.items():
                normalized = normalized.replace(char, ascii_equiv)
            result.normalized_command = normalized

        return score

    def _detect_variable_tricks(self, command: str, result: PreprocessResult) -> float:
        """Detect variable expansion tricks used for obfuscation."""
        score = 0.0

        # Pattern: ${var:offset:length} substring extraction
        substring_pattern = re.compile(r"\$\{[^}]+:[0-9]+:[0-9]+\}")
        matches = substring_pattern.findall(command)
        if len(matches) >= 2:
            result.encodings_detected.append("var_substring")
            result.warnings.append("Variable substring extraction (possible obfuscation)")
            score += 0.3

        # Pattern: ${!var} indirect variable reference
        indirect_pattern = re.compile(r"\$\{![^}]+\}")
        if indirect_pattern.search(command):
            result.encodings_detected.append("var_indirect")
            result.warnings.append("Indirect variable reference detected")
            score += 0.2

        # Pattern: eval with constructed string
        eval_pattern = re.compile(r"eval\s+[\"'\$]")
        if eval_pattern.search(command):
            result.encodings_detected.append("eval")
            result.warnings.append("eval with dynamic content detected")
            score += 0.4

        # Pattern: Concatenated command construction
        # e.g., r='rm'; m='-rf'; $r $m /
        concat_pattern = re.compile(r"(\w+)=['\"][^'\"]+['\"].*\$\1")
        if concat_pattern.search(command):
            result.encodings_detected.append("var_concat")
            result.warnings.append("Variable-based command construction detected")
            score += 0.3

        return score

    def _check_decoded_content(self, result: PreprocessResult) -> None:
        """Check decoded payloads for dangerous content."""
        for payload in result.decoded_payloads:
            # Extract just the decoded part (after the colon)
            if ":" in payload:
                decoded = payload.split(":", 1)[1].strip()
            else:
                decoded = payload

            for pattern in self.DANGEROUS_DECODED_PATTERNS:
                if pattern.search(decoded):
                    result.block_reason = (
                        f"Dangerous command detected in encoded payload: "
                        f"{pattern.pattern}"
                    )
                    return

    def get_risk_factors(self, result: PreprocessResult) -> list[str]:
        """Get human-readable risk factors from the result.

        Args:
            result: The PreprocessResult to analyze.

        Returns:
            List of risk factor descriptions.
        """
        factors = []

        if result.encodings_detected:
            factors.append(
                f"Encoding detected: {', '.join(result.encodings_detected)}"
            )

        if result.decoded_payloads:
            # Sanitize payloads for display
            for payload in result.decoded_payloads[:3]:
                preview = payload[:50] + "..." if len(payload) > 50 else payload
                factors.append(f"Decoded content: {preview}")

        if result.obfuscation_score > 0.3:
            factors.append(
                f"Obfuscation score: {result.obfuscation_score:.2f}"
            )

        return factors
