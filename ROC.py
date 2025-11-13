import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st

# Load the dataset
data = pd.read_csv("heart.csv")

# Split the dataset into features and labels
last_column_header = data.columns[-1]
X = data.drop(columns=[last_column_header])
y = data[last_column_header]

# Split the data into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a dictionary of classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes (Bernoulli)": BernoulliNB(),
    "AdaBoost": AdaBoostClassifier(),
    "Neural Network": MLPClassifier(),
}

# Create a Streamlit app
st.title("Heart Failure Prediction App")

# Display the ROC curves and confusion matrices for each classifier
st.subheader(
    "Receiver Operating Characteristic (ROC) and Confusion Matrix for Multiple Classifiers"
)

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    st.subheader(name)
    st.write(f"ROC AUC: {roc_auc:.2f}")

    y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display the confusion matrix with black text
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Heart Disease", "Heart Disease"])
    plt.yticks(tick_marks, ["No Heart Disease", "Heart Disease"])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    for i in range(2):
        for j in range(2):
            text_color = "k"  # Black text color
            ax.text(
                j,
                i,
                str(conf_matrix[i, j]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=18,
            )

    # Add the color bar
    cbar = ax.figure.colorbar(im, ax=ax)

    st.pyplot(fig)
plt.close()
