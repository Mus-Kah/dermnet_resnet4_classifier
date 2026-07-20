
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    data_dir: Path = Path("DermNet")
    train_dir: Optional[Path] = None   # defaults to data_dir/train
    test_dir: Optional[Path] = None    # defaults to data_dir/test
    img_size: Tuple[int, int] = (380, 380)
    batch_size: int = 16
    seed: int = 42

    save_dir: Path = Path("saved_models/DermNet_EfficientNetB4_v2")
    model_name: str = "DermNet_EfficientNetB4_v2"

    l2_reg: float = 1e-4
    dropout_rate: float = 0.5

    class_weight_min: float = 0.5
    class_weight_max: float = 10.0

    phase1_epochs: int = 50
    phase1_lr: float = 3e-4
    phase1_warmup_epochs: int = 2
    phase1_patience: int = 7
    phase1_focal_gamma: float = 2.0
    phase1_focal_alpha: float = 0.25
    phase1_label_smoothing: float = 0.10

    phase2_epochs: int = 50
    phase2_lr: float = 5e-5
    phase2_unfreeze_fraction: float = 0.40  # top 40% of backbone layers
    phase2_patience: int = 10
    phase2_reduce_lr_patience: int = 4
    phase2_focal_gamma: float = 2.0
    phase2_focal_alpha: float = 0.25
    phase2_label_smoothing: float = 0.08

    phase3_epochs: int = 50
    phase3_lr: float = 1e-5
    phase3_patience: int = 12
    phase3_reduce_lr_patience: int = 5
    phase3_focal_gamma: float = 1.5
    phase3_focal_alpha: float = 0.25
    phase3_label_smoothing: float = 0.05

    weight_decay: float = 1e-5
    weight_decay_phase3: float = 5e-6
    use_mixed_precision: bool = True

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.train_dir = Path(self.train_dir) if self.train_dir else self.data_dir / "train"
        self.test_dir = Path(self.test_dir) if self.test_dir else self.data_dir / "test"
        self.save_dir = Path(self.save_dir)
