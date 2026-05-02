import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import tensorflow as tf

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
Adam = tf.keras.optimizers.Adam

sns.set(style='whitegrid')


def main():
    print("✅ Code started running...")

    # LOAD DATA
    df = pd.read_csv('parkinsons.data')

    print(df.head())

    # DROP name column
    df_processed = df.drop('name', axis=1)

    # ---------------- VISUALS ----------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(col)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(x='status', data=df)
    plt.title('Status Count')
    plt.tight_layout()
    plt.show()

    # ---------------- SPLIT ----------------
    X = df_processed.drop('status', axis=1)
    y = df_processed['status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data ready")

    # =================================================
    # KNN
    # =================================================
    print("\nKNN MODEL")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    knn_pred = knn.predict(X_test_scaled)
    knn_prob = np.array(knn.predict_proba(X_test_scaled))[:, 1]

    print("Accuracy:", accuracy_score(y_test, knn_pred))
    print(classification_report(y_test, knn_pred))

    # Confusion Matrix (FIXED STYLE)
    cm = confusion_matrix(y_test, knn_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                linewidths=1, linecolor='black',
                annot_kws={"size": 14})
    plt.title('KNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # =================================================
    # SVM
    # =================================================
    print("\nSVM MODEL")

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    svm.fit(X_train_scaled, y_train)

    svm_pred = svm.predict(X_test_scaled)
    svm_prob = np.array(svm.predict_proba(X_test_scaled))[:, 1]

    print("Accuracy:", accuracy_score(y_test, svm_pred))
    print(classification_report(y_test, svm_pred))

    cm = confusion_matrix(y_test, svm_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                linewidths=1, linecolor='black',
                annot_kws={"size": 14})
    plt.title('SVM Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # =================================================
    # NEURAL NETWORK
    # =================================================
    print("\nNEURAL NETWORK")

    input_dim = X_train_scaled.shape[1]

    model = Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    nn_prob = model.predict(X_test_scaled).ravel()
    nn_pred = (nn_prob >= 0.5).astype(int)

    print("Accuracy:", accuracy_score(y_test, nn_pred))
    print(classification_report(y_test, nn_pred))

    cm = confusion_matrix(y_test, nn_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                linewidths=1, linecolor='black',
                annot_kws={"size": 14})
    plt.title('Neural Network Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    print("\n FINISHED")


if __name__ == "__main__":
    main()