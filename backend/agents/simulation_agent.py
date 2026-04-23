"""
Simulation Agent
================
Agent 3: Generates the parameter grid for counterfactual simulations.

Responsibilities:
- Build combinatorial parameter space
- Prepare serializable job payloads for parallel execution
"""

from itertools import product
from typing import Any

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger("simulation_agent")
settings = get_settings()


class SimulationAgent:
    """
    Agent 3: Parameter grid generator.

    Generates all combinations of:
    - Entry shifts: [-60, -45, ..., +60] minutes
    - Exit shifts: [-60, -45, ..., +60] minutes
    - Stop-loss levels: [0.25%, 0.5%, 1%, 2%]
    - Position size multipliers: [25%, 50%, 75%, 100%, 150%]

    Total: 9 × 9 × 4 × 5 = 1,620 simulations
    """

    def __init__(self):
        self.name = "SimulationAgent"
        self.entry_shifts = settings.ENTRY_SHIFT_RANGE
        self.exit_shifts = settings.EXIT_SHIFT_RANGE
        self.stop_losses = settings.STOP_LOSS_OPTIONS
        self.size_multipliers = settings.SIZE_MULTIPLIERS

    def generate_param_grid(self, trade: dict) -> list[dict]:
        """
        Generate the full parameter grid for counterfactual simulations.

        Args:
            trade: Validated trade dictionary from IngestionAgent

        Returns:
            List of parameter dictionaries, each ready for simulate_trade()
        """
        logger.info(f"[{self.name}] Generating parameter grid...")

        # --- Compute total combinations ---
        total = (
            len(self.entry_shifts)
            * len(self.exit_shifts)
            * len(self.stop_losses)
            * len(self.size_multipliers)
        )

        logger.info(
            f"[{self.name}] Grid dimensions: "
            f"{len(self.entry_shifts)} entry × "
            f"{len(self.exit_shifts)} exit × "
            f"{len(self.stop_losses)} stop-loss × "
            f"{len(self.size_multipliers)} size = "
            f"{total} total simulations"
        )

        # --- Pre-compute shared fields ---
        original_entry_ts = trade["entry_time"].timestamp()
        original_exit_ts = trade["exit_time"].timestamp()
        direction = trade["direction"]
        original_size = trade["size"]

        # --- Generate all combinations ---
        param_grid = []

        for entry_shift, exit_shift, stop_loss, size_mult in product(
            self.entry_shifts,
            self.exit_shifts,
            self.stop_losses,
            self.size_multipliers,
        ):
            # Skip invalid combinations where shifted exit would be before shifted entry
            shifted_entry = original_entry_ts + (entry_shift * 60)
            shifted_exit = original_exit_ts + (exit_shift * 60)

            if shifted_exit <= shifted_entry:
                continue

            param_grid.append({
                "entry_shift": entry_shift,
                "exit_shift": exit_shift,
                "stop_loss": stop_loss,
                "size_multiplier": size_mult,
                "original_entry_ts": original_entry_ts,
                "original_exit_ts": original_exit_ts,
                "direction": direction,
                "original_size": original_size,
            })

        # --- Log skipped ---
        skipped = total - len(param_grid)
        if skipped > 0:
            logger.info(
                f"[{self.name}] Skipped {skipped} invalid combinations "
                f"(exit before entry after shift)"
            )

        logger.info(f"[{self.name}] ✅ Generated {len(param_grid)} valid parameter sets")

        return param_grid

    def get_grid_stats(self) -> dict:
        """Return statistics about the parameter grid configuration."""
        return {
            "entry_shifts": self.entry_shifts,
            "exit_shifts": self.exit_shifts,
            "stop_losses": self.stop_losses,
            "size_multipliers": self.size_multipliers,
            "total_combinations": (
                len(self.entry_shifts)
                * len(self.exit_shifts)
                * len(self.stop_losses)
                * len(self.size_multipliers)
            ),
        }
