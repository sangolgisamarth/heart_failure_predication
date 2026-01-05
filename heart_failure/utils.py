import sys
import traceback

def log_error(e):
    print("[ERROR]", str(e))
    traceback.print_exc(file=sys.stdout)

def validate_input(data, expected_length):
    if len(data) != expected_length:
        raise ValueError(f"Expected {expected_length} features, got {len(data)}")
