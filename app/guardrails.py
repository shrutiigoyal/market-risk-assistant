"""Input/output guardrails for the risk assistant.

Pure-Python (stdlib only) so the rules are unit-testable without GCP.

Input : length cap -> prompt-injection detection -> PII masking -> topic restriction
Output: PII masking -> system-prompt leak detection -> length cap / empty fallback
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_INPUT_CHARS = 2000
MAX_OUTPUT_CHARS = 4000

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|system)\s+(instructions|rules|prompts?)",
    r"(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"\byou\s+are\s+now\b",
    r"\b(developer|dan|jailbreak)\s+mode\b",
    r"\bact\s+as\s+(if\s+you\s+(are|were)\s+)?(an?\s+)?(unrestricted|unfiltered|different)",
    r"override\s+(your|the)\s+(safety|rules|instructions)",
]
_INJECTION_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]

# --- PII -------------------------------------------------------------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")                 # India PAN
_AADHAAR = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")      # India Aadhaar (12 digits)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3,5}\)?[ -]?)\d{3,4}[ -]?\d{3,4}(?!\d)")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, [pii types found])."""
    found: list[str] = []

    def sub(pattern, label, s, validator=None):
        def repl(m):
            raw = m.group(0)
            if validator and not validator(re.sub(r"\D", "", raw)):
                return raw
            if label not in found:
                found.append(label)
            return f"[{label}]"
        return pattern.sub(repl, s)

    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_PAN, "PAN", text)
    text = sub(_SSN, "SSN", text)
    text = sub(_CARD, "CARD", text, validator=lambda d: 13 <= len(d) <= 19 and _luhn_ok(d))
    text = sub(_AADHAAR, "AADHAAR", text)
    text = sub(_PHONE, "PHONE", text, validator=lambda d: 10 <= len(d) <= 13)
    return text, found


# --- Topic restriction -----------------------------------------------------
ALLOWED_TOPIC_KEYWORDS = {
    "var", "value at risk", "risk", "volcker", "basel", "frtb", "expected shortfall",
    "position", "positions", "pnl", "p&l", "profit", "loss", "desk", "trading", "trade", "portfolio",
    "hedge", "hedging", "market", "regulatory", "regulation", "compliance", "reporting", "stress",
    "backtest", "backtesting", "exposure", "volatility", "confidence", "horizon", "underwriting",
    "market-making", "market making", "proprietary", "liquidity", "limit", "limits", "rates",
    "credit", "equities", "fx", "capital", "dodd-frank", "dodd frank", "metrics",
    "garch", "ewma", "kupiec", "christoffersen", "allocation", "contribute", "contribution", "correlation",
    "diversification", "concentration", "concentrated", "scenario", "shortfall", "asset", "assets", "equity",
    "bond", "bonds", "gold", "forecast", "drawdown", "sharpe", "weights",
}
_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|help|what can you do)\b", re.I)


def is_on_topic(text: str) -> bool:
    if _GREETING.match(text):
        return True
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z&\-]+", lowered))
    if tokens & ALLOWED_TOPIC_KEYWORDS:
        return True
    return any(k in lowered for k in ALLOWED_TOPIC_KEYWORDS if " " in k)


@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    reasons: list[str] = field(default_factory=list)
    pii_found: list[str] = field(default_factory=list)
    user_message: str = ""


def check_input(text: str) -> GuardrailResult:
    text = (text or "").strip()
    if not text:
        return GuardrailResult(False, "", ["empty_input"], user_message="Please type a question.")
    if len(text) > MAX_INPUT_CHARS:
        return GuardrailResult(False, "", ["input_too_long"],
                               user_message=f"Please keep questions under {MAX_INPUT_CHARS} characters.")
    for rx in _INJECTION_RE:
        if rx.search(text):
            return GuardrailResult(False, "", ["prompt_injection"],
                                   user_message="I can't follow instructions that try to change my rules. "
                                                "Ask me about market risk, VaR, or regulatory reporting instead.")
    masked, pii = mask_pii(text)
    if not is_on_topic(masked):
        return GuardrailResult(False, masked, ["off_topic"], pii,
                               user_message="I can only help with market risk, VaR, positions/P&L, and "
                                            "regulatory reporting topics.")
    return GuardrailResult(True, masked, [], pii)


SYSTEM_PROMPT_MARKERS = ("risk_router", "sub_agents", "transfer_to_agent", "system prompt",
                         "portfolio_analyst", "quant_risk_analyst", "research_analyst", "explanation_analyst")


def check_output(text: str) -> GuardrailResult:
    text = (text or "").strip()
    if not text:
        return GuardrailResult(True, "I couldn't produce an answer. Please rephrase your question.", ["empty_output"])
    reasons: list[str] = []
    masked, pii = mask_pii(text)
    if pii:
        reasons.append("pii_in_output")
    if any(m in masked.lower() for m in SYSTEM_PROMPT_MARKERS):
        return GuardrailResult(False, "", ["system_prompt_leak"], pii,
                               user_message="I can't share that. Is there a risk or regulatory question I can help with?")
    if len(masked) > MAX_OUTPUT_CHARS:
        masked = masked[:MAX_OUTPUT_CHARS].rsplit(" ", 1)[0] + " …"
        reasons.append("output_truncated")
    return GuardrailResult(True, masked, reasons, pii)
