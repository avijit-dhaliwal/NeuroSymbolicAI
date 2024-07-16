# Neuro-Symbolic AI for Digit Classification

## Project Overview

This project implements a neuro-symbolic AI system that classifies handwritten digits from the MNIST dataset. It combines the pattern recognition capabilities of neural networks with the interpretability of symbolic reasoning, showcasing an innovative approach to AI that bridges connectionist and symbolic paradigms.

## Detailed Description

### Neural Network Component

The core of the system is a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The network architecture includes:

- Input layer: Accepts 28x28x1 grayscale images
- Two convolutional layers with ReLU activation and max pooling
- Flatten layer to transition from 2D features to 1D
- Two dense layers, including the output layer with softmax activation

This architecture allows the network to learn hierarchical features from the input images, progressively capturing more complex patterns.

### Symbolic Reasoning Component

After classification, a symbolic reasoner applies logical rules to the network's output. It determines properties of the predicted digit, such as:

- Is it even?
- Is it odd?
- Is it prime?

This component demonstrates how traditional AI techniques can augment neural network outputs, providing additional context and interpretability.

### Visualization

The project includes a comprehensive visualization module that offers insights into the AI's decision-making process:

1. Display of the original input image
2. Activation maps for each layer of the CNN
3. Classification probability distribution
4. Graph representation of the symbolic reasoning process
5. Summary of the neuro-symbolic integration results

This visualization serves as a powerful tool for understanding and debugging the AI system's behavior.

## Setup and Usage

### Prerequisites

- Python 3.7+

- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/avijit-dhaliwal/NeuroSymbolicAI.git
cd NeuroSymbolicAI
```
2. Install required packages:
```bash
pip install tensorflow numpy matplotlib seaborn networkx
```
### Running the Project

Execute the main script:
```bash
python main.py
```

This will train the model on the MNIST dataset, perform classification and reasoning on a test image, and generate a visualization saved as 'neuro_symbolic_visualization.png'.

## Code Structure
- `main.py`: Entry point of the application
- `config.py`: Configuration settings for the neural network and training process
- `neuro_symbolic_ai.py`: Core implementation of the NeuroSymbolicAI class
- `visualize.py`: Functions for creating the comprehensive visualization

## How It Works

1. Data Loading: The MNIST dataset is loaded and preprocessed.

2. Model Training: A CNN is trained on the MNIST data, learning to recognize digits.

3. Neuro-Symbolic Integration:
   - The trained CNN classifies a test image.
   - The symbolic reasoner applies logical rules to the classification result.

4. Visualization:
   - Layer activations are extracted from the CNN.
   - A multi-part visualization is generated, showing the original image, layer activations, classification probabilities, and symbolic reasoning results.

This process demonstrates the synergy between neural networks and symbolic AI, combining the strengths of both approaches.

## Extending the Project
Potential areas for expansion include:
- Implementing more complex symbolic reasoning rules
- Exploring bidirectional integration between neural and symbolic components
- Applying the neuro-symbolic approach to other datasets or problem domains

## Contributing
Contributions to enhance the project are welcome. Please fork the repository and submit a pull request with your proposed changes.

## License
[MIT License](LICENSE)