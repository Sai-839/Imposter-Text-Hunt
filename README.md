#  Impostor Text Hunt - Hybrid Ensemble Pipeline 🕵️‍♂️

This project implements a sophisticated hybrid model to detect AI-generated or "impostor" text for the [Fake or Real - The Impostor Hunt](https://www.kaggle.com/competitions/fake-or-real-the-impostor-hunt) Kaggle competition.

The solution uses a stacking ensemble approach, combining predictions from classical machine learning models, a fine-tuned Transformer, and an embedding-based dissimilarity score to achieve robust performance. The entire pipeline, from data setup to submission file generation, is contained within a single notebook.

---

## 🚀 Key Features

* **Hybrid Modeling**: Leverages the strengths of different modeling paradigms.
* **Classical ML Ensemble**: A `VotingClassifier` combines **RandomForest**, **GradientBoosting**, **SVC**, **XGBoost**, and **CatBoost** on engineered linguistic features.
* **Transformer Power**: Fine-tunes a `distilbert-base-uncased` model to understand deep semantic context between text pairs.
* **Embedding Dissimilarity**: Uses `SentenceTransformer` to calculate the cosine dissimilarity between text embeddings, providing a powerful signal for how different the two texts are.
* **Robust Validation**: Employs `StratifiedKFold` cross-validation to prevent data leakage and ensure the model generalizes well.
* **Stacking Ensemble**: A `LogisticRegression` meta-learner combines the predictions from the three base models to make a final, more accurate prediction.
* **Automated Pipeline**: Handles package installation, Kaggle data download, preprocessing, training, and prediction in one go.

---

## ⚙️ Model Architecture

The model uses a two-level stacking architecture.

**Level 1: Base Models**
Three distinct models generate initial predictions (probabilities) for each text pair:

1.  **Classical Feature Ensemble**:
    * **Input**: 18 linguistic features are extracted from each pair of texts (character count, word count, type-token ratio, punctuation, etc.).
    * **Model**: A `VotingClassifier` that averages probabilities from five tuned classical models.
2.  **Fine-Tuned Transformer**:
    * **Input**: The raw text pair (`text_1`, `text_2`).
    * **Model**: A `distilbert-base-uncased` model fine-tuned on the task of classifying the text pair.
3.  **Embedding Dissimilarity Score**:
    * **Input**: The raw text pair (`text_1`, `text_2`).
    * **Model**: An `all-MiniLM-L6-v2` Sentence Transformer encodes each text into an embedding. The dissimilarity is calculated as `1 - cosine_similarity(emb1, emb2)`.

**Level 2: Meta-Learner (Stacking)**
The out-of-fold predictions from the three base models are used as features to train a final meta-learner.

* **Input**: A feature set with 3 columns: `[classical_prob, transformer_prob, embedding_prob]`.
* **Model**: A `LogisticRegression` classifier that learns the optimal weights for combining the base model predictions.
* **Output**: The final prediction used for the submission file.



---

## 🛠️ Setup and Installation

### 1. Prerequisites
You need Python 3.x and the following libraries. The script will attempt to install them automatically if the first cell is uncommented.

* `peft`, `accelerate`, `transformers`, `datasets`
* `catboost`, `xgboost`, `scikit-learn`
* `pandas`, `numpy`, `scipy`
* `nltk`, `sentence-transformers`
* `kaggle`, `google-colab` (for file handling in Colab)

You can install them manually using pip:
```bash
pip install peft accelerate transformers datasets catboost xgboost scikit-learn pandas numpy scipy nltk sentence-transformers kaggle
````

### 2\. Kaggle API Key

This project requires a Kaggle API key to download the competition dataset.

1.  Go to your Kaggle account page, find the "API" section, and click "Create New Token". This will download a `kaggle.json` file.
2.  When you run the notebook, it will prompt you to upload this `kaggle.json` file.

-----

## ▶️ How to Run

1.  **Environment**: This notebook is designed to run in a GPU-enabled environment like Google Colab or a Kaggle Kernel for optimal performance, especially for fine-tuning the Transformer model.
2.  **Upload `kaggle.json`**: Run the first few cells. When prompted, upload the `kaggle.json` file you downloaded.
3.  **Execute All Cells**: Run the entire notebook from top to bottom. The script will:
      * Configure the Kaggle API.
      * Download and unzip the competition data.
      * Preprocess the text and engineer features.
      * Run the full K-Fold cross-validation and training pipeline.
      * Generate predictions on the test set.
      * Create a `submission.csv` file in the correct format.

-----

## 📋 Workflow Breakdown

1.  **Configuration**: Key parameters like `SEED`, `MODEL_NAME`, and `NUM_FOLDS` are set. The device (`cuda` or `cpu`) is automatically detected.
2.  **Data Setup**: The script securely configures the Kaggle API and downloads the competition data into a `./data` directory.
3.  **Preprocessing**: Text is cleaned (lowercase, remove punctuation/digits) and lemmatized. A suite of linguistic features is extracted.
4.  **K-Fold Training Loop**:
      * The data is split into `N` folds.
      * For each fold, the three base models are trained on the training portion and make predictions on the validation portion.
      * These validation predictions (Out-of-Fold predictions) are stored.
      * The trained models from each fold are saved.
5.  **Meta-Learner Training**: The complete set of Out-of-Fold (OOF) predictions is used to train the final `LogisticRegression` meta-learner. This ensures the meta-learner is trained on data it has never seen before.
6.  **Inference and Submission**:
      * For the test set, predictions are made using the saved models from all `N` folds.
      * The predictions are averaged across the folds to produce a stable final prediction.
      * The final probabilities are converted into the required `real_text_id` labels and saved to `submission.csv`.

<!-- end list -->

```
```
