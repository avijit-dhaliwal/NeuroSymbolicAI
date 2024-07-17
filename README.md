# Advanced Neuro-Symbolic AI for Digit Classification

## Project Overview
This project implements a sophisticated neuro-symbolic AI system that combines deep learning with symbolic reasoning to classify and analyze handwritten digits from the MNIST dataset. It demonstrates the synergy between neural networks and traditional AI approaches, offering both powerful pattern recognition and interpretable logical reasoning.

## Key Features
- Convolutional Neural Network (CNN) for image classification
- Advanced symbolic reasoning with multiple mathematical and logical rules
- Dynamic integration of neural network outputs with symbolic processing
- Comprehensive visualization of the AI's decision-making process

## Detailed Description

### Neural Network Component
The core of the system is a CNN built with TensorFlow/Keras, featuring:
- Convolutional and pooling layers for feature extraction
- Dense layers for classification
- Softmax activation for multi-class prediction

### Symbolic Reasoning Component
The symbolic reasoner applies a variety of rules to the network's output, including:
- Mathematical properties (e.g., even/odd, prime, perfect square, Fibonacci)
- Digit-specific attributes (e.g., number of closed loops, symmetry)
- Contextual analysis (consistency with previous predictions)
- Meta-reasoning (rule complexity, certainty estimation)

### Neuro-Symbolic Integration
The system combines neural network predictions with symbolic reasoning to provide a rich, interpretable output that goes beyond simple classification.

### Visualization
A comprehensive visualization module offers insights into the AI's decision-making:
- Original input image display
- CNN layer activation maps
- Classification probability distribution
- Symbolic reasoning process representation
- Integrated neuro-symbolic results summary

## Setup and Usage

1. Clone the repository:
```bash
git clone https://github.com/avijit-dhaliwal/NeuroSymbolicAI.git
cd advanced-neuro-symbolic-ai
```

2. Install required packages:
```bash
pip install tensorflow numpy matplotlib seaborn networkx
```
3. Run the main script:
```bash
python main.py
```
## Code Structure
- `main.py`: Entry point and orchestration of the AI system
- `config.py`: Configuration settings
- `neuro_symbolic_ai.py`: Implementation of the NeuroSymbolicAI class
- `visualize.py`: Visualization functions

## Future Directions
- Implement bidirectional information flow between neural and symbolic components
- Extend to multi-modal data analysis
- Explore applications in other domains (e.g., natural language processing, scientific discovery)

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes or improvements.

## License
[MIT License](LICENSE)