# Breast cancer prediction

The Breast Cancer Prediction API is a Flask-based web service that provides a simple interface to predict whether a given set of features is indicative of benign or malignant breast cancer. The API utilizes a machine learning model trained on a breast cancer dataset to make predictions.

## Machine learning model
The predictive model is built using the Random Forest Classifier, trained on the well-known Breast Cancer Wisconsin (Diagnostic) Dataset. The model has undergone rigorous testing and validation to ensure reliable predictions.

### Testing and training the model
The testing phase is a critical step in evaluating the performance of the breast cancer prediction model based on Random Forest classification. The process involves the following key steps:

* Identification of Highly Correlated Features:

    Features with high correlation are identified using a specified threshold. This step aims to reduce redundant data and enhance the model's efficiency.

* Testing with Different Criteria, Estimators, and Thresholds:

    The model is rigorously tested with varying criteria (Gini and Entropy), the number of estimators, and different correlation thresholds, providing insights into the impact on model performance and aiding in feature optimization

* Performance Metrics and Labels:

    Accuracy is measured for each test configuration, offering a quantitative assessment of the model's predictive capabilities. Labels such as the criterion used, the number of estimators, the features selection threshold, and the number of selected features are recorded.
* Results Visualization (Optional):

    Results can be visualized to provide a clear overview of how different criteria, estimators, and features selection thresholds influence model accuracy.
* Execution Time Tracking:

    The duration of the testing phase is recorded, contributing to an understanding of the model's computational efficiency.

 Performance Visualization
The scatter plot visualizes the performance of the breast cancer prediction model under different configurations. Key factors influencing the model, such as the choice of criterion (Gini or Entropy), the number of estimators, and the features selection threshold, are represented. The x-axis reflects the number of estimators, the y-axis displays the accuracy achieved, and the color and size variations indicate the criterion and threshold, respectively.

### Test Configuration Visualization
The scatter plot provides a visual representation of the breast cancer prediction model's performance across various test configurations. Each point on the plot corresponds to a specific set of parameters, including the choice of criterion (Gini or Entropy), the number of estimators, and the features selection threshold. The x-axis represents the number of estimators, the y-axis represents the achieved accuracy, and the color and size of the points indicate the criterion and threshold, respectively.


<div align="center">
  <img src="./images/tests_performance.png" alt="Confusion Matrix" width="800"/>
</div>


### Confusion Matrix Evaluation
The confusion matrix plot provides a concise snapshot of our breast cancer prediction model's performance in distinguishing between Benign (B) and Malignant (M) cases. It showcases True Positives, True Negatives, False Positives, and False Negatives, offering a quick assessment of the model's accuracy in classifying different types of breast cancer. This visual aid is instrumental in gauging the model's reliability and precision in clinical scenarios.


<div align="center">
  <img src="./images/confusion_matrix.png" alt="Confusion Matrix" />
</div>

## Installation

**1** -  Clone this repository to your machine by running

```bash
cd https://github.com/hamza-hadj-aissa/breast_cancer_prediction.git
```

**2** - Navigate to the project's directory
```
cd breast_cancer_prediction/
```

**3** - Set Up a Virtual Environment
```
python -m venv env
```

**4** - Activate the Virtual Environment
```
source env/bin/activate
```

**5** - Install required libraries
```bash
pip install -r requirements.txt
```

**6** - Run the Flask API
```bash
python main.py
```
This will server the API on : http://localhost:8000


## API Reference

#### Predict cancer

```http
  POST /predict
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `radius_mean` | `float` | **Required**. Radius of the mean nuclei |
| `texture_mean` | `float` | **Required**.Texture of the mean nuclei |
| `smoothness_mean` | `float` | **Required**. Smoothness of the mean nuclei|
| `symmetry_mean` | `float` | **Required**.  Symmetry of the mean nuclei|
| `fractal_dimension_mean` | `float` | **Required**. Fractal dimension of the mean nuclei |
| `texture_se` | `float` | **Required**. Standard error of texture |
| `smoothness_se` | `float` | **Required**. Standard error of smoothness |
| `symmetry_se` | `float` | **Required**. Standard error of symmetry |

* Response is a JSON object. Example: 
```
{
  "prediction": [
    {"Benign": 0.8},
    {"Malignant": 0.2}
  ]
}
```





## Next.js UI Repository

The Next.js user interface for breast cancer prediction is available on GitHub. You can find the source code and project details in the following repository:

- **Repository:** [breast_cancer_prediction Web App](https://github.com/hamza-hadj-aissa/breast_cancer_prediction_ui)

### Getting Started

To integrate the Next.js UI with the Breast Cancer Prediction API, follow these steps:

1. Clone the Next.js UI repository:

```bash
git clone https://github.com/hamza-hadj-aissa/breast_cancer_prediction_ui.git
```

2. Navigate to the project directory:
```
cd breast_cancer_prediction_ui
```
3. Install dependencies:
```
npm install
```
4. Run the Next.js app:
```
npm run dev
```
This will start the development server, and you can access the app at http://localhost:3000.










