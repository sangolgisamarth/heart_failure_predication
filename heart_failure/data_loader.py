import pandas as pd

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load_data(self):
        try:
            data = pd.read_csv(self.path)
            print(f"[INFO] Data loaded with shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.path}")
            return None
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            return None
