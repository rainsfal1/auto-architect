<p align="center">
  <a href="https://nust.edu.pk/"><img width="200" height="200" src="resources/nust.svg"/></a> &nbsp;&nbsp;&nbsp;&nbsp; <a href="https://seecs.nust.edu.pk/"><img width="200" height="200" src="resources/seecs.png"/></a>
</p>

# Auto-Architect: Neural Architecture Search for NER Models

## Overview

Auto-Architect is a Neural Architecture Search (NAS) project designed to automatically find the best architecture for Named Entity Recognition (NER) models. The project utilizes the NNI (Neural Network Intelligence) framework to explore various neural network architectures based on a predefined search space, optimizing them for performance on NER tasks.

## Features

- **Dynamic Search Space**: Define and customize the search space for hyperparameters like embedding dimensions, CNN filters, LSTM hidden sizes, etc.
- **Automated Architecture Search**: Use NNI's NAS capabilities to discover the optimal architecture for NER models.
- **Model Evaluation**: Evaluate the performance of each architecture using predefined metrics.
- **Result Export**: Save the best-performing architectures for further analysis or deployment.

## Project Structure
```
auto-architect/
├── .gitignore
├── main.py
├── data/
│   ├── __init__.py
│   ├── embeddings/
│   ├── processed/
│   └── processing.py
├── model_results/
│   └── model_config.json
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   └── ner_model.py
│   └── nas/
│       ├── __init__.py
│       ├── training.py
│       └── search_space.json
├── utils/
│   ├── __init__.py
│   └── metrics.py
└── config.py
```


## Getting Started

### Prerequisites

- Python 3.8 or higher
- NNI (Neural Network Intelligence)
- Required Python packages listed in `requirements.txt`

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/rainsfal1/auto-architect.git
    cd auto-architect
    ```

2. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1. **Define the Search Space:**

   - Edit the `src/nas/search_space.json` file to customize the hyperparameters and their ranges.

2. **Adjust NAS Configurations:**

   - Modify the `src/config.py` file to set NAS experiment configurations such as trial concurrency, maximum trial number, etc.

### Running the Project

1. **Prepare the Data:**

   - Ensure that your data is processed and available in the `data/processed` directory.

2. **Run the NAS Experiment:**

    ```bash
    python main.py
    ```

3. **View Results:**

   - Check the `model_results/` directory for the exported best model architectures.

## Files and Their Purpose

- `main.py`: Main script to run the NAS experiment. Initializes the NAS, evaluates models, and exports results.
- `src/nas/search_space.json`: JSON file defining the search space for NAS.
- `data/processing.py`: Contains functions to create data loaders and preprocess data.
- `src/models/ner_model.py`: Defines the model space for NER tasks.
- `src/config.py`: Configuration settings for NAS experiments.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [NNI](https://github.com/microsoft/nni) for providing the NAS framework.
- [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/) for deep learning tools.
- Any other relevant acknowledgements.



