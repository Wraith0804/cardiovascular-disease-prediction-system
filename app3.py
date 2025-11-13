import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def generate_learning_curves(models, X_train, y_train):
    learning_curve_data = {}
    for name, model in models.items():  # Use .items() to iterate through the dictionary
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=10,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
        )

        learning_curve_data[name] = {
            "train_sizes": train_sizes,
            "train_scores_mean": np.mean(train_scores, axis=1),
            "test_scores_mean": np.mean(test_scores, axis=1),
        }

    return learning_curve_data


# To ignore all warnings
warnings.filterwarnings("ignore")

# To set the title
st.title("Diabetes Prediction and Model Deployment")

# Sidebar for user interaction
st.sidebar.header("Data Loading")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File", type=["csv"]
)  # To upload CSV file - Comma Separated Values File
if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Reading the uploaded csv
    st.sidebar.success(
        "File successfully uploaded!"
    )  # Message printed after successful upload of csv

# Data Exploration
st.sidebar.header("Data Exploration")
if st.sidebar.checkbox(
    "Show Dataset"
):  # Creating checkbox for Displaying Dataset in Sidebar
    if "df" in locals():  # Checking presence of dataset in local namespace
        st.write(df.head(10))  # Prints first 10 rows of the dataset
    else:  # If no dataset uploaded
        st.warning("Please upload a dataset first.")

if st.sidebar.checkbox(
    "Show Summary Statistics"
):  # To summarize dataset - Checkbox created in sidebar
    if "df" in locals():
        st.write(df.describe())
    else:
        st.warning("Please upload a dataset first.")

# Data Preprocessing
st.sidebar.header("Data Preprocessing")  # Starting Data Preprocessing
if st.sidebar.checkbox(
    "Data Preprocessing"
):  # Checkbox (in sidebar) to initiate Data preprocessing
    if "df" in locals():
        last_column_header = df.columns[-1]  # Access the last column header
        X = df.drop(columns=[last_column_header])
        # Defining X variables of dataset - Removal of Y variable
        y = df[last_column_header]  # Defining Y variable of dataset - "CLASS" feature

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )  # Splitting data in to test size of 20% --> implies training size of 80%
        # Random_state used to control randomness factor - Can reproduce same results again and fixes randomness of test and train

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Standardization method - Scaling data to fit a standard normal distribution.
        # Standard normal distribution - Mean of 0 and standard deviation of 1

        pca = PCA(0.95)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        # Used to reduce dimensionality of data from  3D to 2D
        # 0.95 Argument passed --> Represents desired explained variance - Retain enough principal components to explain 95% of data.

        smoteenn = SMOTEENN(random_state=42)
        X_train_resampled, y_train_resampled = smoteenn.fit_resample(
            X_train_pca, y_train
        )
        # Implementation of SMOTE and Edited Nearest Neighbour
        # Addresses the class imbalance problem by oversampling the minority class and then cleaning data
        X_test_resampled, y_test_resampled = smoteenn.fit_resample(X_test_pca, y_test)
        st.success("Data preprocessing completed!")

st.sidebar.header("Model Training and Evaluation")
if st.sidebar.checkbox("Model Training and Evaluation"):
    if "X_train_resampled" in locals():
        st.subheader("Select Models")
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes (Bernoulli)": BernoulliNB(),
            "AdaBoost": AdaBoostClassifier(),
            "Neural Network": MLPClassifier(),
        }

        selected_models = st.multiselect(
            "Select models for evaluation", list(models.keys())
        )

        if st.button("Evaluate Models"):
            scores = {}
            for model_name, model in models.items():
                if model_name in selected_models:
                    model.fit(X_train_resampled, y_train_resampled)
                    y_pred = model.predict(X_test_resampled)
                    accuracy = accuracy_score(y_test_resampled, y_pred)
                    scores[model_name] = accuracy
            st.subheader("Model Evaluation Results")
            for model_name, accuracy in scores.items():
                st.write(f"{model_name}: Accuracy = {accuracy:.2f}")

        # Model Evaluation and Metrics
        if st.sidebar.checkbox("Model Evaluation and Metrics"):
            if "X_train_resampled" in locals():
                st.subheader("Select Models")
                # ... (same code for model selection)

                if st.button("Evaluate Models and Metrics"):
                    model_metrics = {}
                    for model_name, model in models.items():
                        if model_name in selected_models:
                            model.fit(X_train_resampled, y_train_resampled)
                            y_pred = model.predict(X_test_resampled)
                            accuracy = accuracy_score(y_test_resampled, y_pred)
                            precision = precision_score(y_test_resampled, y_pred)
                            recall = recall_score(y_test_resampled, y_pred)
                            f1 = f1_score(y_test_resampled, y_pred)
                            roc_auc = roc_auc_score(y_test_resampled, y_pred)

                            model_metrics[model_name] = {
                                "Accuracy": accuracy,
                                "Precision": precision,
                                "Recall": recall,
                                "F1 Score": f1,
                                "ROC AUC": roc_auc,
                            }

                    st.subheader("Model Metrics")
                    for model_name, metrics in model_metrics.items():
                        st.write(f"{model_name}:")
                        st.markdown(f"&emsp;&emsp; Accuracy: {metrics['Accuracy']:.4f}")
                        st.markdown(
                            f"&emsp;&emsp; Precision: {metrics['Precision']:.4f}"
                        )
                        st.markdown(f"&emsp;&emsp; Recall: {metrics['Recall']:.4f}")
                        st.markdown(f"&emsp;&emsp; F1 Score: {metrics['F1 Score']:.4f}")
                        st.markdown(f"&emsp;&emsp; ROC AUC: {metrics['ROC AUC']:.4f}")

                        fig, ax = plt.subplots()
                        metrics_names = list(metrics.keys())
                        metrics_values = list(metrics.values())
                        ax.bar(
                            metrics_names,
                            metrics_values,
                            color=["blue", "green", "orange", "red", "purple"],
                        )
                        plt.xticks(rotation=15)
                        plt.title(f"{model_name} Metrics")
                        st.pyplot(fig)

                        fpr, tpr, thresholds = roc_curve(y_test_resampled, y_pred)
                        plt.figure()
                        plt.plot(fpr, tpr, color="darkorange", lw=2)
                        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.title(f"ROC Curve for {model_name}")
                        st.pyplot(plt)

                    learning_curve_data = generate_learning_curves(
                        models, X_train_resampled, y_train_resampled
                    )

                    # Display learning curves
                    st.subheader("Learning Curves")
                    for name, data in learning_curve_data.items():
                        plt.figure()
                        plt.plot(
                            data["train_sizes"],
                            data["train_scores_mean"],
                            label="Train Accuracy",
                        )
                        plt.plot(
                            data["train_sizes"],
                            data["test_scores_mean"],
                            label="Validation Accuracy",
                        )
                        plt.xlabel("Training Dataset Size")
                        plt.ylabel("Accuracy")
                        plt.title(f"Learning Curve - {name}")
                        plt.legend()
                        st.pyplot(plt)

        # Ensemble Evaluation
        st.subheader("Ensemble Evaluation")
        ensemble_models = {
            "Voting Classifier": VotingClassifier(
                estimators=[
                    (model_name, model)
                    for model_name, model in models.items()
                    if model_name in selected_models
                ],
                voting="hard",
            )
        }
        selected_ensemble_models = st.multiselect(
            "Select models for ensemble", list(ensemble_models.keys())
        )

        if st.button("Evaluate Ensemble"):
            ensemble_scores = {}
            for ensemble_name, ensemble_model in ensemble_models.items():
                if ensemble_name in selected_ensemble_models:
                    ensemble_model.fit(X_train_resampled, y_train_resampled)
                    y_pred_ensemble = ensemble_model.predict(X_test_resampled)
                    accuracy_ensemble = accuracy_score(
                        y_test_resampled, y_pred_ensemble
                    )
                    ensemble_scores[ensemble_name] = accuracy_ensemble
            st.subheader("Ensemble Model Evaluation Results")
            for ensemble_name, accuracy in ensemble_scores.items():
                st.write(f"{ensemble_name}: Accuracy = {accuracy:.2f}")

    else:
        st.warning("Please complete data preprocessing first.")


# Hyperparameter Tuning - Formative
st.sidebar.header("Hyperparameter Tuning")
if st.sidebar.checkbox("Hyperparameter Tuning"):
    if "X_train_resampled" in locals():
        st.subheader("Select Model for Hyperparameter Tuning")
        hyperparameter_models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes (Bernoulli)": BernoulliNB(),
            "AdaBoost": AdaBoostClassifier(),
        }

        selected_hyperparameter_model = st.selectbox(
            "Select a model for hyperparameter tuning",
            list(hyperparameter_models.keys()),
        )

        if st.button("Tune Hyperparameters"):
            model = hyperparameter_models[selected_hyperparameter_model]
            hyperparameters = {}
            if selected_hyperparameter_model == "Logistic Regression":
                hyperparameters = {
                    "C": [0.001, 0.01, 0.1, 1, 10],
                    "penalty": ["l1", "l2", "elasticnet", "none"],
                    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                }
            elif selected_hyperparameter_model == "Decision Tree":
                hyperparameters = {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30, 40, 50],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                }
            # Add hyperparameters for other models here

            clf = GridSearchCV(model, hyperparameters, cv=10)
            clf.fit(X_train_resampled, y_train_resampled)

            st.subheader(f"Best Hyperparameters for {selected_hyperparameter_model}")
            st.write(clf.best_params_)

    else:
        st.warning("Please complete data preprocessing first.")

# Model Deployment
st.sidebar.header("Model Deployment")
if st.sidebar.checkbox("Model Deployment"):
    if "X_train_resampled" in locals():
        st.subheader("Choose a Model for Deployment")
        deployment_models = {
            "Voting Classifier": VotingClassifier(
                estimators=[
                    ("Decision Tree", DecisionTreeClassifier()),
                    ("Random Forest", RandomForestClassifier()),
                    ("SVM", SVC()),
                    ("K-Nearest Neighbors", KNeighborsClassifier()),
                    ("AdaBoost", AdaBoostClassifier()),
                ],
                voting="hard",
            )
        }

        selected_deployment_model = st.selectbox(
            "Select a model for deployment", list(deployment_models.keys())
        )

        if st.button("Deploy Model"):
            model = deployment_models[selected_deployment_model]
            model.fit(X_train_resampled, y_train_resampled)

            # Save the model to a file
            model_filename = "diabetes_prediction_model.pkl"
            joblib.dump(model, model_filename)

            st.success(
                f"Model '{selected_deployment_model}' deployed and saved as '{model_filename}'"
            )

    else:
        st.warning("Please complete data preprocessing first.")

# Render the Streamlit app
if "df" in locals():
    st.write("### Loaded Dataset:")
    st.write(df.head(10))
else:
    st.write("### Upload a CSV dataset to get started.")
