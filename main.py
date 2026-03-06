import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

from src.data_loader import load_dataset
from src.preprocessing import clean_data, convert_labels, sample_dataset, split_features
from src.rf_model import train_random_forest
from src.dl_model import build_dl_model
from src.hybrid_model import hybrid_predict


DATA_PATH = "data"


def main():

    # Load dataset
    data = load_dataset(DATA_PATH)

    # Clean dataset
    data = clean_data(data)

    # Convert labels
    data = convert_labels(data)

    # Random sampling
    data = sample_dataset(data, fraction=0.2)

    # Split features
    X, y = split_features(data)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    # Feature scaling
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    rf = train_random_forest(X_train, y_train)

    # Prepare DL input
    X_train_dl = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_dl = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build DL model
    dl_model = build_dl_model((X_train.shape[1], 1))

    # Train DL model
    dl_model.fit(
        X_train_dl[:100000],
        y_train[:100000],
        epochs=5,
        batch_size=256,
        validation_split=0.2
    )

    # Hybrid Prediction
    predictions = hybrid_predict(rf, dl_model, X_test, X_test_dl)

    # Evaluation
    print("Hybrid Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification Report\n")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()