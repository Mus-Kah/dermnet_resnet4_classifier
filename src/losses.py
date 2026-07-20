"""Custom loss functions."""
import tensorflow as tf
from tensorflow import keras


class FocalLoss(keras.losses.Loss):

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 label_smoothing: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / y_true.shape[-1]

        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1 - y_pred, self.gamma)
        focal_loss = weight * ce

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return config
