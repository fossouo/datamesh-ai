"""
Redactor - Data redaction and masking engine.

Applies redaction rules to data based on classification,
policies, and user context.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any
from enum import Enum

logger = logging.getLogger(__name__)


class MaskingMethod(Enum):
    """Data masking methods."""
    FULL = "full"           # Replace entire value with ***
    PARTIAL = "partial"     # Show first/last characters
    HASH = "hash"           # Replace with hash
    MASK_LAST4 = "mask_last4"  # Show only last 4 characters
    MASK_FIRST4 = "mask_first4"  # Show only first 4 characters
    EMAIL = "email"         # Mask email keeping domain
    PHONE = "phone"         # Mask phone keeping country code
    NULL = "null"           # Replace with null


@dataclass
class RedactionRule:
    """Redaction rule definition."""
    field: str
    method: MaskingMethod
    reason: str = ""
    policy_ref: str = ""


class Redactor:
    """
    Data redaction engine.

    Applies masking and redaction rules to protect sensitive data
    based on classification and policies.
    """

    def __init__(self):
        # Default redaction rules by classification
        self._classification_rules = {
            "PII": MaskingMethod.PARTIAL,
            "SECRET": MaskingMethod.FULL,
            "SENSITIVE": MaskingMethod.PARTIAL,
            "INTERNAL": MaskingMethod.PARTIAL,
        }

        # Field-specific rules
        self._field_rules = {
            "email": MaskingMethod.EMAIL,
            "phone": MaskingMethod.PHONE,
            "ssn": MaskingMethod.FULL,
            "social_security": MaskingMethod.FULL,
            "card_number": MaskingMethod.MASK_LAST4,
            "credit_card": MaskingMethod.MASK_LAST4,
            "password": MaskingMethod.FULL,
            "secret": MaskingMethod.FULL,
            "token": MaskingMethod.FULL,
            "api_key": MaskingMethod.FULL,
        }

    async def redact(
        self,
        records: list[dict],
        classifications: dict[str, list[str]],
        user: dict,
        policies: list[str],
    ) -> tuple[list[dict], dict]:
        """
        Apply redaction to records.

        Args:
            records: List of data records to redact
            classifications: Field classifications {field: [classifications]}
            user: User context
            policies: Applied policies

        Returns:
            Tuple of (redacted_records, redaction_report)
        """
        logger.info(f"Redacting {len(records)} records")

        # Build redaction rules
        rules = self._build_rules(classifications, user, policies)

        # Apply redaction
        redacted_records = []
        redaction_counts = {}

        for record in records:
            redacted_record = self._redact_record(record, rules, redaction_counts)
            redacted_records.append(redacted_record)

        # Build report
        report = {
            "recordsProcessed": len(records),
            "fieldsRedacted": redaction_counts,
            "rulesApplied": [
                {
                    "field": r.field,
                    "method": r.method.value,
                    "reason": r.reason,
                }
                for r in rules
            ],
        }

        return redacted_records, report

    def _build_rules(
        self,
        classifications: dict[str, list[str]],
        user: dict,
        policies: list[str],
    ) -> list[RedactionRule]:
        """Build redaction rules from classifications and policies."""
        rules = []
        user_roles = user.get("roles", [])

        # Admin users may have reduced redaction
        is_admin = "admin" in user_roles

        for field, field_classifications in classifications.items():
            # Skip if user is admin (unless SECRET)
            if is_admin and "SECRET" not in field_classifications:
                continue

            # Determine masking method
            method = self._determine_method(field, field_classifications)

            rules.append(RedactionRule(
                field=field,
                method=method,
                reason=f"Classifications: {', '.join(field_classifications)}",
                policy_ref=policies[0] if policies else "",
            ))

        # Add field-specific rules for common patterns
        for pattern, method in self._field_rules.items():
            if not any(r.field == pattern for r in rules):
                rules.append(RedactionRule(
                    field=pattern,
                    method=method,
                    reason=f"Field pattern: {pattern}",
                ))

        return rules

    def _determine_method(
        self,
        field: str,
        classifications: list[str],
    ) -> MaskingMethod:
        """Determine masking method for a field."""
        field_lower = field.lower()

        # Check field-specific rules first
        for pattern, method in self._field_rules.items():
            if pattern in field_lower:
                return method

        # Use classification-based rules
        if "SECRET" in classifications:
            return MaskingMethod.FULL
        if "PII" in classifications:
            return MaskingMethod.PARTIAL
        if "SENSITIVE" in classifications:
            return MaskingMethod.PARTIAL

        return MaskingMethod.PARTIAL

    def _redact_record(
        self,
        record: dict,
        rules: list[RedactionRule],
        counts: dict,
    ) -> dict:
        """Apply redaction rules to a single record."""
        redacted = record.copy()

        for key, value in record.items():
            # Find applicable rule
            rule = self._find_rule(key, rules)

            if rule and value is not None:
                redacted[key] = self._apply_mask(value, rule.method)
                counts[key] = counts.get(key, 0) + 1

        return redacted

    def _find_rule(self, field: str, rules: list[RedactionRule]) -> RedactionRule | None:
        """Find applicable rule for a field."""
        field_lower = field.lower()

        for rule in rules:
            if rule.field.lower() == field_lower:
                return rule
            if rule.field.lower() in field_lower:
                return rule

        return None

    def _apply_mask(self, value: Any, method: MaskingMethod) -> Any:
        """Apply masking method to a value."""
        if value is None:
            return None

        str_value = str(value)

        if method == MaskingMethod.FULL:
            return "***REDACTED***"

        if method == MaskingMethod.NULL:
            return None

        if method == MaskingMethod.PARTIAL:
            if len(str_value) <= 4:
                return "****"
            return str_value[:2] + "*" * (len(str_value) - 4) + str_value[-2:]

        if method == MaskingMethod.MASK_LAST4:
            if len(str_value) <= 4:
                return str_value
            return "*" * (len(str_value) - 4) + str_value[-4:]

        if method == MaskingMethod.MASK_FIRST4:
            if len(str_value) <= 4:
                return str_value
            return str_value[:4] + "*" * (len(str_value) - 4)

        if method == MaskingMethod.EMAIL:
            # Mask email keeping domain
            if "@" in str_value:
                parts = str_value.split("@")
                masked_local = parts[0][:2] + "***"
                return f"{masked_local}@{parts[1]}"
            return "***@***"

        if method == MaskingMethod.PHONE:
            # Keep first 3 digits, mask rest
            digits = re.sub(r'\D', '', str_value)
            if len(digits) > 3:
                return digits[:3] + "-***-****"
            return "***-***-****"

        if method == MaskingMethod.HASH:
            import hashlib
            return hashlib.sha256(str_value.encode()).hexdigest()[:16]

        return str_value
