"""
Pattern Detection Agent
========================
Agent 6: Identifies behavioral mistakes by comparing actual trade vs optimal.

Responsibilities:
- Tag behavioral patterns: Early Entry, Late Exit, Stop Too Tight, etc.
- Assign severity levels
- Provide human-readable explanations

Rules Engine:
- Early Entry:   actual entry > 15min before optimal
- Late Entry:    actual entry > 15min after optimal
- Early Exit:    actual exit > 15min before optimal (missed profit)
- Late Exit:     actual exit > 15min after optimal (overstayed)
- Stop Too Tight: actual stop-loss < optimal by >50%
- Stop Too Loose: no stop-loss used or very wide
- Over-sizing:   actual size > optimal by >25%
- Under-sizing:  actual size < optimal by >50%
"""

from typing import Any

from backend.utils.logger import get_logger

logger = get_logger("pattern_agent")


class PatternDetectionAgent:
    """
    Agent 6: Behavioral pattern detector.

    Compares the actual trade against the best counterfactual result
    to identify systematic trading mistakes.
    """

    # Thresholds for pattern detection
    ENTRY_THRESHOLD_MINUTES = 15
    EXIT_THRESHOLD_MINUTES = 15
    SIZE_OVER_THRESHOLD = 0.25      # 25% over optimal
    SIZE_UNDER_THRESHOLD = 0.50     # 50% under optimal
    STOP_TIGHT_THRESHOLD = 0.50     # 50% tighter than optimal

    def __init__(self):
        self.name = "PatternDetectionAgent"

    def detect_patterns(
        self,
        actual_trade: dict,
        best_trade: dict,
        aggregated: dict,
    ) -> list[dict]:
        """
        Detect behavioral patterns by comparing actual vs optimal trade.

        Args:
            actual_trade: The original trade as executed
            best_trade: The best counterfactual simulation result
            aggregated: Full aggregated output (for metric-based patterns)

        Returns:
            List of pattern dictionaries with tag, detail, and severity
        """
        logger.info(f"[{self.name}] Analyzing trade patterns...")

        patterns = []

        if best_trade is None:
            logger.warning(f"[{self.name}] No best trade found, skipping pattern detection")
            return patterns

        # ============================================
        # ENTRY TIMING PATTERNS
        # ============================================
        entry_shift = best_trade.get("entry_shift", 0)

        if entry_shift < -self.ENTRY_THRESHOLD_MINUTES:
            # Optimal entry was EARLIER → trader entered too late
            patterns.append({
                "tag": "LATE_ENTRY",
                "detail": (
                    f"You entered {abs(entry_shift)} minutes too late. "
                    f"The optimal entry was {abs(entry_shift)} minutes earlier. "
                    f"This suggests hesitation or waiting for excessive confirmation."
                ),
                "severity": "high" if abs(entry_shift) > 30 else "medium",
            })

        elif entry_shift > self.ENTRY_THRESHOLD_MINUTES:
            # Optimal entry was LATER → trader entered too early
            patterns.append({
                "tag": "EARLY_ENTRY",
                "detail": (
                    f"You entered {entry_shift} minutes too early. "
                    f"The optimal entry was {entry_shift} minutes later. "
                    f"This suggests impulsive entry without proper setup confirmation."
                ),
                "severity": "high" if entry_shift > 30 else "medium",
            })

        # ============================================
        # EXIT TIMING PATTERNS
        # ============================================
        exit_shift = best_trade.get("exit_shift", 0)

        if exit_shift < -self.EXIT_THRESHOLD_MINUTES:
            # Optimal exit was EARLIER → trader held too long
            patterns.append({
                "tag": "LATE_EXIT",
                "detail": (
                    f"You exited {abs(exit_shift)} minutes too late. "
                    f"The optimal exit was {abs(exit_shift)} minutes earlier. "
                    f"You may have been anchored to unrealistic targets or "
                    f"affected by disposition effect (holding losers too long)."
                ),
                "severity": "high" if abs(exit_shift) > 30 else "medium",
            })

        elif exit_shift > self.EXIT_THRESHOLD_MINUTES:
            # Optimal exit was LATER → trader exited too early
            patterns.append({
                "tag": "EARLY_EXIT",
                "detail": (
                    f"You exited {exit_shift} minutes too early. "
                    f"The optimal exit was {exit_shift} minutes later. "
                    f"This suggests profit-taking anxiety — you cut winners too short."
                ),
                "severity": "high" if exit_shift > 30 else "medium",
            })

        # ============================================
        # STOP-LOSS PATTERNS
        # ============================================
        optimal_stop = best_trade.get("stop_loss", 0.01)

        # Compare with the default stop-loss (1%)
        default_stop = 0.01

        if optimal_stop > default_stop * (1 + self.STOP_TIGHT_THRESHOLD):
            # Optimal stop was WIDER → trader's stop was too tight
            patterns.append({
                "tag": "STOP_TOO_TIGHT",
                "detail": (
                    f"The optimal stop-loss was {optimal_stop*100:.2f}%, "
                    f"which is significantly wider than typical. "
                    f"A tighter stop may have caused premature exit due to normal volatility."
                ),
                "severity": "medium",
            })

        if best_trade.get("exit_reason") == "stop_loss":
            patterns.append({
                "tag": "STOP_LOSS_OPTIMAL",
                "detail": (
                    f"Even the best counterfactual hit its stop-loss at "
                    f"{optimal_stop*100:.2f}%. This suggests the trade setup "
                    f"may have been fundamentally flawed or the market moved against."
                ),
                "severity": "high",
            })

        # ============================================
        # POSITION SIZING PATTERNS
        # ============================================
        optimal_size_mult = best_trade.get("size_multiplier", 1.0)

        if optimal_size_mult < 1.0 - self.SIZE_UNDER_THRESHOLD:
            # Optimal was smaller → trader over-sized
            patterns.append({
                "tag": "OVER_SIZING",
                "detail": (
                    f"The optimal position size was {optimal_size_mult*100:.0f}% "
                    f"of your actual size. You were over-exposed, increasing "
                    f"drawdown risk without proportional reward."
                ),
                "severity": "high" if optimal_size_mult < 0.5 else "medium",
            })

        elif optimal_size_mult > 1.0 + self.SIZE_OVER_THRESHOLD:
            # Optimal was larger → trader under-sized
            patterns.append({
                "tag": "UNDER_SIZING",
                "detail": (
                    f"The optimal position size was {optimal_size_mult*100:.0f}% "
                    f"of your actual size. You could have captured more profit "
                    f"with a larger position given the favorable setup."
                ),
                "severity": "low",
            })

        # ============================================
        # OUTCOME-BASED PATTERNS
        # ============================================
        metrics = aggregated.get("metrics", {})

        # Check if majority of alternatives were profitable
        profitable_pct = metrics.get("profitable_pct", 50)

        if profitable_pct < 30:
            patterns.append({
                "tag": "BAD_SETUP",
                "detail": (
                    f"Only {profitable_pct:.0f}% of counterfactual scenarios were profitable. "
                    f"This suggests the trade setup itself was unfavorable — "
                    f"no timing or sizing adjustment could reliably save it."
                ),
                "severity": "high",
            })
        elif profitable_pct > 80 and metrics.get("actual_pnl", 0) <= 0:
            patterns.append({
                "tag": "MISSED_OPPORTUNITY",
                "detail": (
                    f"{profitable_pct:.0f}% of scenarios were profitable, "
                    f"yet your actual trade lost money. "
                    f"The setup was strong but execution was poor."
                ),
                "severity": "high",
            })

        # ============================================
        # SUMMARY
        # ============================================
        logger.info(
            f"[{self.name}] ✅ Detected {len(patterns)} behavioral patterns: "
            f"{[p['tag'] for p in patterns]}"
        )

        return patterns
