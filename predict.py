import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# Load trained files
model = load_model("ids_model.keras")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


def predict_file(filepath):

    # Read uploaded CSV
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove columns used only during training
    df = df.drop(["Flow ID", "Timestamp"], axis=1, errors="ignore")

    # Remove label if present
    if "Label" in df.columns:
        df = df.drop("Label", axis=1)

    # Replace invalid values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Scale data
    X = scaler.transform(df)

    # CNN input shape
    X = X.reshape(X.shape[0], 6, 13, 1)

    # Predict
    predictions = model.predict(X)

    predicted_class = np.argmax(predictions, axis=1)

    labels = label_encoder.inverse_transform(predicted_class)

    df["Prediction"] = labels

    return df
