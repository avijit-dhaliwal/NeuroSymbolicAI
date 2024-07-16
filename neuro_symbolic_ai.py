import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from config import Config

class SymbolicReasoner:
    def __init__(self):
        self.rules = {
            'is_even': lambda x: x % 2 == 0,
            'is_odd': lambda x: x % 2 != 0,
            'is_prime': self._is_prime
        }

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def apply_rules(self, number: int) -> Dict[str, bool]:
        return {rule: func(number) for rule, func in self.rules.items()}

class NeuroSymbolicAI:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.symbolic_reasoner = SymbolicReasoner()
        self.intermediate_model = None

    def _create_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.config.input_shape)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.config.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = self._create_model()
        self.model.fit(x_train, y_train, epochs=self.config.epochs, 
                       batch_size=self.config.batch_size, validation_split=self.config.validation_split)
        
        # Create intermediate model for layer outputs
        self.intermediate_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[layer.output for layer in self.model.layers[1:]]  # Exclude input layer
        )

    def predict_and_reason(self, image: np.ndarray) -> Tuple[int, Dict[str, bool]]:
        prediction = self.model.predict(image[np.newaxis, ...])
        predicted_class = np.argmax(prediction[0])
        reasoning_results = self.symbolic_reasoner.apply_rules(predicted_class)
        return predicted_class, reasoning_results

    def get_layer_outputs(self, image: np.ndarray) -> List[np.ndarray]:
        return self.intermediate_model.predict(image[np.newaxis, ...])