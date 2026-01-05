import pickle
from heart_failure.model import HeartFailureANN
from heart_failure.preprocess import Preprocessor
from heart_failure.data_loader import DataLoader

# 1. Load data
data_path = "notebook/data/heart_failure_clinical_records_dataset.csv"
data = DataLoader(data_path).load_data()

# 2. Preprocess data
preprocessor = Preprocessor()
X_train, X_test, y_train, y_test, scaler = preprocessor.preprocess(data)

# 3. Train model
model = HeartFailureANN(input_dim=X_train.shape[1])
history = model.train(X_train, y_train)

# 4. Save model & scaler
model.save_model("models/my_keras_model.h5")
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("[INFO] Training complete. Model and scaler saved to 'models/' folder.")
