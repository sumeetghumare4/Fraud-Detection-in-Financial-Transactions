# Fraud Detection in Financial Transactions

## Overview

This project focuses on building a machine learning model to detect fraudulent transactions in financial datasets. Fraud detection is a critical task in the finance industry, as it helps prevent significant financial losses and ensures the security of financial transactions.

The model is built using deep learning techniques, specifically a neural network implemented with Keras and scikit-learn for model evaluation and hyperparameter tuning. The model is trained and validated on historical transaction data, aiming to accurately classify transactions as either fraudulent or legitimate.


## Getting Started

### Prerequisites

- Python 3.8 or higher
- Conda (optional, for managing the environment)
- pip

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/fraud-detection.git
    cd fraud-detection
    ```

2. **Create and activate a virtual environment:**

    - Using `conda`:

      ```bash
      conda env create -f environment.yml
      conda activate fraud-detection
      ```

    - Using `pip`:

      ```bash
      python3 -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -r requirements.txt
      ```

3. **Download and prepare the data:**

   Place the dataset files in the `data/` directory. Ensure that the file names match the expected format (e.g., `X_train.csv`, `y_train.csv`).

### Usage

1. **Exploratory Data Analysis (EDA):**

   Navigate to the `notebooks/` directory and open the `exploratory_analysis.ipynb` notebook to explore the dataset and understand the distribution of features and labels.

2. **Model Training:**

   The model can be trained using the `model_training.ipynb` notebook or by running the `train_model.py` script:

   ```bash
   python scripts/train_model.py


