import tensorflow as tf
import numpy as np
import math
from typing import Dict, Any, List, Tuple
from config import Config

class SymbolicReasoner:
    def __init__(self):
        self.rules = {
            'is_even': lambda x: x % 2 == 0,
            'is_odd': lambda x: x % 2 != 0,
            'is_prime': self._is_prime,
            'is_perfect_square': self._is_perfect_square,
            'is_fibonacci': self._is_fibonacci,
            'num_closed_loops': self._num_closed_loops,
            'is_symmetric': self._is_symmetric,
            'greater_than_5': lambda x: x > 5,
        }
        self.meta_rules = {
            'certainty': self._calculate_certainty,
            'rule_complexity': self._rule_complexity
        }
        self.context = {'previous_predictions': []}

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def _is_perfect_square(self, n: int) -> bool:
        return int(math.sqrt(n)) ** 2 == n

    def _is_fibonacci(self, n: int) -> bool:
        phi = 0.5 + 0.5 * math.sqrt(5.0)
        a = phi * n
        return n == 0 or abs(round(a) - a) < 1.0 / n

    def _num_closed_loops(self, n: int) -> int:
        loops = {0: 1, 6: 1, 8: 2, 9: 1}
        return loops.get(n, 0)

    def _is_symmetric(self, n: int) -> bool:
        return n in [0, 1, 3, 8]

    def _calculate_certainty(self, results: Dict[str, Any]) -> float:
        return sum(1 for v in results.values() if isinstance(v, bool) and v) / len(results)

    def _rule_complexity(self, results: Dict[str, Any]) -> Dict[str, str]:
        complexity = {
            'is_even': 'simple',
            'is_odd': 'simple',
            'is_prime': 'moderate',
            'is_perfect_square': 'moderate',
            'is_fibonacci': 'complex',
            'num_closed_loops': 'moderate',
            'is_symmetric': 'simple',
            'greater_than_5': 'simple'
        }
        return {rule: complexity[rule] for rule in results if rule in complexity}

    def apply_rules(self, number: int, probabilities: List[float]) -> Dict[str, Any]:
        results = {rule: func(number) for rule, func in self.rules.items()}
        
        results['consistent_with_previous'] = self._check_consistency(number)
        results['confidence'] = probabilities[number]
        results['certainty'] = self.meta_rules['certainty'](results)
        results['rule_complexity'] = self.meta_rules['rule_complexity'](results)
        
        self.context['previous_predictions'].append(number)
        if len(self.context['previous_predictions']) > 5:
            self.context['previous_predictions'].pop(0)
        
        return results

    def _check_consistency(self, number: int) -> bool:
        if not self.context['previous_predictions']:
            return True
        return abs(number - self.context['previous_predictions'][-1]) <= 2

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
            inputs=self.model.inputs,
            outputs=[layer.output for layer in self.model.layers[1:]]  # Exclude input layer
        )

    def predict_and_reason(self, image: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        prediction = self.model.predict(image[np.newaxis, ...])
        predicted_class = np.argmax(prediction[0])
        reasoning_results = self.symbolic_reasoner.apply_rules(predicted_class, prediction[0])
        return predicted_class, reasoning_results

    def get_layer_outputs(self, image: np.ndarray) -> List[np.ndarray]:
        return self.intermediate_model.predict(image[np.newaxis, ...])