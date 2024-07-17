import tensorflow as tf
import numpy as np
from neuro_symbolic_ai import NeuroSymbolicAI
from visualize import visualize_neuro_symbolic_ai
from config import Config
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        config = Config()
        
        logging.info("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
        
        logging.info("Creating and training the neuro-symbolic AI system...")
        neuro_symbolic_system = NeuroSymbolicAI(config)
        neuro_symbolic_system.train(x_train, y_train)
        
        logging.info("Testing the system...")
        test_image = x_test[0]
        predicted_class, reasoning_results = neuro_symbolic_system.predict_and_reason(test_image)
        
        logging.info(f"Predicted class: {predicted_class}")
        logging.info("Symbolic reasoning results:")
        for key, value in reasoning_results.items():
            logging.info(f"  {key}: {value}")
        
        logging.info("Generating visualization...")
        model_outputs = neuro_symbolic_system.get_layer_outputs(test_image)
        visualize_neuro_symbolic_ai(model_outputs, test_image, predicted_class, reasoning_results)
        
        logging.info("Process completed successfully. Visualization saved as 'neuro_symbolic_visualization.png'")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()