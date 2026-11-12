"""LoRA training: SFT warm start (``sft_trainer``, loss ``losses.sft_ce``), then DMPO (``dmpo_trainer``, loss
``losses.dmpo``).

Importing this package does not import torch; the trainers do.
"""

from geniac_baseline.trainer.config import DMPOConfig, TrainConfig, WandbConfig, load_config
from geniac_baseline.trainer.loss import LossInputs, LossOutput, compute_loss

__all__ = ["DMPOConfig", "LossInputs", "LossOutput", "TrainConfig", "WandbConfig", "compute_loss", "load_config"]
