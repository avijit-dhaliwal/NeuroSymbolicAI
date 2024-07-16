from dataclasses import dataclass

@dataclass
class Config:
    input_shape: tuple = (28, 28, 1)
    num_classes: int = 10
    epochs: int = 5
    batch_size: int = 32
    validation_split: float = 0.2