from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import callbacks
import numpy as np

class HeartFailureANN:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(16, activation='relu', input_dim=self.input_dim))
        model.add(Dense(8, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(4, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, epochs=500, batch_size=32, validation_split=0.2):
        early_stopping = callbacks.EarlyStopping(
            min_delta=0.001,
            patience=20,
            restore_best_weights=True
        )
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
        return history

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return (y_pred > 0.5).astype(int)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        from keras.models import load_model
        self.model = load_model(model_path)
