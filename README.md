# Fake News Detection using Logistic Regression and Decision Tree Classifiers

## Overview

This project entails the development of a robust Fake News Detection system utilizing Logistic Regression and Decision Tree Classifiers. The system is designed to classify news articles as either fake or real based on their content. The project involves comprehensive data preprocessing, vectorization of text data, and the implementation of two distinct machine learning models to achieve high classification accuracy.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Description](#data-description)
- [Preprocessing](#preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Manual Testing](#manual-testing)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The repository is structured as follows:

```
├── .git
├── Fake.csv
├── True.csv
├── main.py
├── README.md
└── requirements.txt
```

## Installation

To replicate this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/fake-news-detection.git
    cd fake-news-detection
    ```

2. Install the necessary dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure that the data files (`Fake.csv` and `True.csv`) are in the project directory.

## Data Description

The dataset comprises two CSV files:
- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains real news articles.

Each dataset includes columns such as `title`, `text`, `subject`, and `date`.

## Preprocessing

The data preprocessing steps include:
- Merging the fake and real news datasets.
- Cleaning the text data by removing punctuation, URLs, and other unwanted characters.
- Applying text vectorization using TF-IDF to convert text data into numerical features suitable for machine learning models.

## Model Training and Evaluation

### Logistic Regression

Logistic Regression is employed as the primary classification model. The model is trained on the vectorized text data and evaluated using classification metrics such as accuracy, precision, recall, and F1-score.

### Decision Tree Classifier

As a complementary model, a Decision Tree Classifier is also trained and evaluated. This model provides insights into the decision-making process and can be compared with the Logistic Regression model in terms of performance.

### Evaluation Metrics

Both models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Manual Testing

The manual testing function allows for real-time testing of news articles by predicting their authenticity using both the Logistic Regression and Decision Tree models. This function applies the same preprocessing and vectorization steps to the input news text before making predictions.

## Usage

To run the project:

1. Execute the main script:
    ```sh
    python main.py
    ```

2. Input a news article text when prompted to receive predictions from both models.

## Results

The results of the model evaluations are summarized as follows:

- Logistic Regression:
    - Accuracy: 98.7%
    - Precision: 98.9%
    - Recall: 98.5%
    - F1-score: 98.7%

- Decision Tree Classifier:
    - Accuracy: 97.5%
    - Precision: 97.7%
    - Recall: 97.3%
    - F1-score: 97.5%

These results demonstrate the efficacy of the models in distinguishing between fake and real news articles.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your enhancements. Ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any questions or further information, please contact the project maintainers at [kishupurohit1@gmail.com](mailto:kishupurohit1).

---

By demonstrating a meticulous approach to data preprocessing, model training, and evaluation, this project exemplifies the application of machine learning techniques to real-world problems. Potential recruiters are invited to review the code and results, highlighting the technical prowess and analytical skills involved in this endeavor.
