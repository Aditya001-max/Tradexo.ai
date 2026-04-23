"""
LLM Agent (Hugging Face)
========================
Agent 7: Generates behavioral coaching using Hugging Face LLMs.

Responsibilities:
- Build structured prompts from trade data + patterns
- Call Hugging Face Inference API (Mistral / Llama)
- Handle rate limits, retries, and fallbacks
- Return coaching text

Models (in priority order):
1. mistralai/Mistral-7B-Instruct-v0.3
2. meta-llama/Meta-Llama-3-8B-Instruct
"""

import time
from typing import Optional

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger("llm_agent")
settings = get_settings()

# Models in priority order
MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


class LLMAgent:
    """
    Agent 7: LLM-powered trading coach.

    Uses Hugging Face Inference API to generate personalized
    behavioral coaching based on detected patterns.
    """

    def __init__(self):
        self.name = "LLMAgent"
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Hugging Face InferenceClient."""
        token = settings.HF_API_TOKEN
        if not token or token == "hf_your_token_here":
            logger.warning(
                f"[{self.name}] No HF_API_TOKEN set. "
                f"LLM coaching will use rule-based fallback."
            )
            return

        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=token)
            logger.info(f"[{self.name}] Hugging Face client initialized")
        except ImportError:
            logger.error(
                f"[{self.name}] huggingface_hub not installed. "
                f"Run: pip install huggingface-hub"
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize HF client: {e}")

    def _build_prompt(
        self,
        trade: dict,
        patterns: list[dict],
        best_trade: dict,
        metrics: dict,
    ) -> str:
        """
        Build a structured coaching prompt.

        Args:
            trade: Actual trade details
            patterns: Detected behavioral patterns
            best_trade: Best counterfactual trade
            metrics: Summary metrics

        Returns:
            Formatted prompt string
        """
        # --- Format patterns ---
        pattern_text = "\n".join(
            f"  - {p['tag']} ({p['severity']}): {p['detail']}"
            for p in patterns
        ) if patterns else "  - No significant patterns detected."

        # --- Build prompt ---
        prompt = f"""You are an expert trading coach and behavioral finance analyst.

A trader has submitted a trade for counterfactual analysis. Based on the data below,
provide specific, actionable coaching feedback.

## ACTUAL TRADE
- Asset: {trade.get('asset', 'Unknown')}
- Direction: {trade.get('direction', 'Unknown')}
- Size: {trade.get('size', 0)} units
- Duration: {trade.get('duration_minutes', 0)} minutes
- Actual P&L: ${metrics.get('actual_pnl', 0):.2f}

## OPTIMAL ALTERNATIVE
- Entry shift: {best_trade.get('entry_shift', 0)} minutes (negative = earlier)
- Exit shift: {best_trade.get('exit_shift', 0)} minutes (negative = earlier)
- Stop-loss: {best_trade.get('stop_loss', 0)*100:.2f}%
- Position size: {best_trade.get('size_multiplier', 1)*100:.0f}% of actual
- Optimal P&L: ${best_trade.get('pnl', 0):.2f}
- Improvement: {best_trade.get('improvement', 'N/A')}

## SIMULATION METRICS
- Total simulations: {metrics.get('total_simulations', 0)}
- Profitable scenarios: {metrics.get('profitable_pct', 0):.0f}%
- Best P&L: ${metrics.get('best_pnl', 0):.2f}
- Worst P&L: ${metrics.get('worst_pnl', 0):.2f}
- Median P&L: ${metrics.get('median_pnl', 0):.2f}

## DETECTED BEHAVIORAL PATTERNS
{pattern_text}

## YOUR TASK
Provide coaching in this exact format:

**What Happened:** [Brief factual summary of the trade and outcome]

**Key Mistake:** [The single most impactful mistake identified]

**Optimal Action:** [What the trader should have done differently]

**Behavioral Pattern:** [The psychological pattern behind the mistake]

**One Rule to Remember:** [A single, memorable rule the trader can apply next time]

Keep your response concise (under 250 words), specific to this trade, and actionable.
Do NOT use generic advice. Reference the specific numbers from the analysis."""

        return prompt

    def generate_coaching(
        self,
        trade: dict,
        patterns: list[dict],
        best_trade: dict,
        metrics: dict,
    ) -> str:
        """
        Generate coaching insights using HuggingFace LLM.

        Falls back to rule-based coaching if API is unavailable.

        Args:
            trade: Actual trade details
            patterns: Detected behavioral patterns
            best_trade: Best counterfactual trade
            metrics: Summary metrics

        Returns:
            Coaching text string
        """
        logger.info(f"[{self.name}] Generating coaching insights...")

        prompt = self._build_prompt(trade, patterns, best_trade, metrics)

        # --- Try LLM ---
        if self.client:
            for model in MODELS:
                result = self._call_llm(model, prompt)
                if result:
                    logger.info(f"[{self.name}] ✅ Coaching generated via {model}")
                    return result

        # --- Fallback to rule-based ---
        logger.info(f"[{self.name}] Using rule-based coaching fallback")
        return self._rule_based_coaching(trade, patterns, best_trade, metrics)

    def _call_llm(
        self,
        model: str,
        prompt: str,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Call a Hugging Face model with retry logic.

        Args:
            model: Model identifier
            prompt: The coaching prompt
            max_retries: Number of retry attempts

        Returns:
            Generated text or None on failure
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"[{self.name}] Calling {model} (attempt {attempt + 1})")

                response = self.client.chat_completion(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional trading coach. "
                                "Provide specific, data-driven feedback. "
                                "Be concise and actionable."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0.7,
                )

                text = response.choices[0].message.content
                if text and len(text.strip()) > 50:
                    return text.strip()

                logger.warning(f"[{self.name}] Empty response from {model}")

            except Exception as e:
                wait = 2 ** attempt  # Exponential backoff
                logger.warning(
                    f"[{self.name}] {model} call failed (attempt {attempt + 1}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        logger.error(f"[{self.name}] All retries exhausted for {model}")
        return None

    def _rule_based_coaching(
        self,
        trade: dict,
        patterns: list[dict],
        best_trade: dict,
        metrics: dict,
    ) -> str:
        """
        Generate coaching text using rules (no LLM needed).

        This is the fallback when HuggingFace API is unavailable.
        """
        lines = []

        # --- What Happened ---
        actual_pnl = metrics.get("actual_pnl", 0)
        outcome = "profit" if actual_pnl > 0 else "loss"
        lines.append(
            f"**What Happened:** You took a {trade.get('direction', 'unknown')} "
            f"position in {trade.get('asset', 'Unknown')} for "
            f"{trade.get('duration_minutes', 0)} minutes, resulting in a "
            f"${actual_pnl:.2f} {outcome}."
        )

        # --- Key Mistake ---
        if patterns:
            worst = max(patterns, key=lambda p: {"high": 3, "medium": 2, "low": 1}.get(p["severity"], 0))
            lines.append(f"\n**Key Mistake:** {worst['detail']}")
        else:
            lines.append("\n**Key Mistake:** No significant mistakes detected — your execution was close to optimal.")

        # --- Optimal Action ---
        best_pnl = best_trade.get("pnl", 0)
        improvement = best_trade.get("improvement", "N/A")
        lines.append(
            f"\n**Optimal Action:** The best scenario involved entering "
            f"{abs(best_trade.get('entry_shift', 0))} minutes "
            f"{'earlier' if best_trade.get('entry_shift', 0) < 0 else 'later'}, "
            f"exiting {abs(best_trade.get('exit_shift', 0))} minutes "
            f"{'earlier' if best_trade.get('exit_shift', 0) < 0 else 'later'}, "
            f"with a {best_trade.get('stop_loss', 0.01)*100:.2f}% stop-loss "
            f"at {best_trade.get('size_multiplier', 1)*100:.0f}% position size. "
            f"This would have yielded ${best_pnl:.2f} ({improvement})."
        )

        # --- Behavioral Pattern ---
        if patterns:
            tags = [p["tag"] for p in patterns]
            if "EARLY_ENTRY" in tags:
                lines.append(
                    "\n**Behavioral Pattern:** Impulsive entry — you tend to "
                    "enter trades before the setup fully confirms. This is often "
                    "driven by FOMO (Fear of Missing Out)."
                )
            elif "LATE_EXIT" in tags:
                lines.append(
                    "\n**Behavioral Pattern:** Disposition effect — you hold "
                    "positions too long, either hoping losers will recover or "
                    "anchoring to unrealistic profit targets."
                )
            elif "OVER_SIZING" in tags:
                lines.append(
                    "\n**Behavioral Pattern:** Overconfidence — you risk too much "
                    "capital per trade, which amplifies both gains and losses "
                    "beyond what the setup warrants."
                )
            else:
                lines.append(
                    f"\n**Behavioral Pattern:** {patterns[0]['tag'].replace('_', ' ').title()} — "
                    f"{patterns[0]['detail']}"
                )
        else:
            lines.append("\n**Behavioral Pattern:** Your execution was disciplined. Keep it up.")

        # --- One Rule ---
        profitable_pct = metrics.get("profitable_pct", 50)
        if profitable_pct < 40:
            lines.append(
                "\n**One Rule to Remember:** Before entering, ask: "
                "'Would 70% of timing variations be profitable?' If not, skip the trade."
            )
        elif any(p["tag"] == "EARLY_ENTRY" for p in patterns):
            lines.append(
                "\n**One Rule to Remember:** Wait for one more confirmation candle "
                "before entering. Patience beats precision."
            )
        elif any(p["tag"] in ("LATE_EXIT", "EARLY_EXIT") for p in patterns):
            lines.append(
                "\n**One Rule to Remember:** Set your exit before entry. "
                "Pre-commit to a time or price target and don't override it mid-trade."
            )
        else:
            lines.append(
                "\n**One Rule to Remember:** Review your top 3 alternatives "
                "after every trade. Pattern recognition improves execution over time."
            )

        return "\n".join(lines)
