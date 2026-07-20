"""Environment setup helpers (TensorFlow/GPU/mixed precision)."""
import tensorflow as tf


def setup_environment(use_mixed_precision: bool = True) -> list:
    """Print TF/GPU info and enable mixed precision if a GPU is present."""
    print(f"TensorFlow: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU(s): {gpus if gpus else 'None – running on CPU'}")

    if gpus and use_mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision: float16 enabled")

    return gpus
