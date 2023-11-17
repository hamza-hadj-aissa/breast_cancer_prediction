import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time

# Record start time
start_time = time.time()

# load dataset
print("Loading dataset...")
dataset_df = pd.read_csv("./breast-cancer.csv")
# drop id of records, as it is not needed
dataset_df.drop(columns=["id"], inplace=True)

print("Transforming...")
# transform categorical data
le = LabelEncoder()
dataset_df['diagnosis'] = le.fit_transform(dataset_df['diagnosis'])
dataset_df.sample(10)

# transform continuous data
std_sc = StandardScaler()
dataset_df.iloc[:, 1:] = std_sc.fit_transform(dataset_df.iloc[:, 1:])
dataset_df.head()


X = dataset_df.drop("diagnosis", axis=1)
Y = dataset_df["diagnosis"]

print("Splitting training and test data...")
# split train and test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.75, random_state=0)
correlation_matrix = X_train.corr()

# plot the heatmap of the correlation between features
# plt.figure(figsize=(8, 7))
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5)
# plt.show()


# identify high correlated features
# the highest ones compared to given threshold
# are dropped to reduce redundant data
def get_high_correlated_features(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # compare the correlation value between i and j with the value of the threshold
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col_corr.add(corr_matrix.columns[i])
    return col_corr


def visualize_results(criterion_labels, estimator_labels, threshold_labels, accuracies):
    results_df = pd.DataFrame({
        'Criterion': criterion_labels,
        'n_estimators': estimator_labels,
        'Threshold': threshold_labels,
        'Accuracy': accuracies
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='n_estimators', y='Accuracy', hue='Criterion',
                    size='Threshold', data=results_df, palette='viridis', sizes=(20, 200))
    plt.title('Breast cancer classification Performance')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.legend(title='Criterion')
    plt.show()


def start_test():
    accuracies = []
    criterion_labels = []
    estimator_labels = []
    threshold_labels = []
    number_of_features = []
    counter = 0
    # test by gini and entropy criterion
    criterions = ["gini", "entropy"]
    test_start_time = time.time()
    for criterion in criterions:
        # test by number of estimators
        estimators = [100, 200, 300, 500, 1000]
        for estimator in estimators:
            # test by features selection threshold
            thresholds = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            for threshold in thresholds:
                nbr_of_remaining_tests = len(
                    criterions)*len(estimators)*len(thresholds) - counter
                print(f"Remaining tests: {nbr_of_remaining_tests}")
                counter = counter + 1
                print(
                    "--------------------------------------------------------------------")
                print(
                    f"Testing for:\n Criterion: {criterion} - Number of estimators: {estimator} - Tresholds: {threshold}")
                # Store labels (test criterias)
                criterion_labels.append(criterion)
                estimator_labels.append(estimator)
                threshold_labels.append(threshold)

                corr_features = get_high_correlated_features(
                    X_train, threshold)
                # print("Number of highly correlated features:", len(corr_features))

                # store number of features as a label
                number_of_features.append(len(corr_features))

                X_train.drop(corr_features, axis=1)
                X_test.drop(corr_features, axis=1)

                model = RandomForestClassifier(
                    n_estimators=estimator, criterion=criterion, random_state=0)
                model.fit(X_train, Y_train)

                # Make predictions
                Y_pred = model.predict(X_test)

                # Evaluate accuracy
                accuracy = accuracy_score(Y_pred, Y_test)
                # store accuracy as a label
                accuracies.append(accuracy)

                print(f"Accuracy: {accuracy}")

                print(
                    "--------------------------------------------------------------------")
    # visualize_results(criterion_labels, estimator_labels,
    #                   threshold_labels, accuracies)
    test_end_time = time.time()
    print(f"Finished testing in: {test_end_time - test_start_time} seconds")
    return {"criterion_labels": criterion_labels,
            "estimator_labels": estimator_labels,
            "threshold_labels": threshold_labels,
            "accuracies": accuracies,
            "number_of_features": number_of_features
            }


def get_model(X_train, X_test, Y_train, Y_test, criterion, estimator, threshold):
    # features selection
    corr_features = get_high_correlated_features(
        X_train, threshold)
    X_train.drop(corr_features, axis=1)
    X_test.drop(corr_features, axis=1)

    model = RandomForestClassifier(
        n_estimators=estimator, criterion=criterion, random_state=0)
    model.fit(X_train, Y_train)

    # Make predictions
    Y_pred = model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(Y_pred, Y_test)

    # Display classification report
    print("Classification Report:")
    print(classification_report(Y_pred, Y_test))

    # Display confusion matrix
    conf_matrix = confusion_matrix(Y_pred, Y_test)
    print("Confusion Matrix:")
    print(conf_matrix)

    return [model, corr_features]


print("Testing model trainings...")
model_test_results = start_test()

model_test_results_df = pd.DataFrame({
    'criterion_labels': model_test_results["criterion_labels"],
    'n_estimators': model_test_results["estimator_labels"],
    'threshold': model_test_results["threshold_labels"],
    "number_of_features": model_test_results["number_of_features"],
    'accuracy': model_test_results["accuracies"],
})

# sort test results
model_test_results_sorted_df = model_test_results_df.sort_values(by=['accuracy', 'number_of_features', 'n_estimators'],
                                                                 ascending=[False, True, True])

# get the best criterias for our model
first_row = model_test_results_sorted_df.iloc[0]

criterion = first_row["criterion_labels"]
n_estimator = first_row["n_estimators"]
threshold = first_row["threshold"]
number_of_features = first_row["number_of_features"]
accuracy = first_row["accuracy"]

print("Training the best model...")
print("Best criterias :")
print(f"Criterion: {criterion}")
print(f"Number of estimators: {n_estimator}")
print(f"Threshold: {threshold}")
print(f"Number of features: {number_of_features}")
print(f"Selected features: {
      list(get_high_correlated_features(X_train, threshold))}")
# Selected features: ['concave points_mean', 'perimeter_se', 'area_mean', 'texture_worst', 'area_se', 'perimeter_mean', 'area_worst', 'radius_worst', 'concave points_worst', 'perimeter_worst']
print(f"Accuracy: {accuracy}")
model_training_start_time = time.time()
model = get_model(X_train, X_test, Y_train, Y_test,
                  criterion, n_estimator, threshold)
model_training_end_time = time.time()

print(
    f"Finished training model in: {model_training_end_time - model_training_start_time} seconds")

model_test_results_df.to_csv('./model_test_results.csv')
# Save the model using joblib
model_filename = "breast_cancer_prediction_model.joblib"
joblib.dump(model, model_filename)
print(f"Model saved to ./{model_filename}")


end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
