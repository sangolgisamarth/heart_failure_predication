from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess(self, data, target_col="DEATH_EVENT", test_size=0.25, random_state=7):
        try:
            X = data.drop(target_col, axis=1)
            y = data[target_col]

            X_scaled = self.scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            return X_train, X_test, y_train, y_test, self.scaler
        except KeyError:
            raise ValueError(f"Target column {target_col} not found in data")
        except Exception as e:
            raise e
