import os
import sys

# Allow imports from src/ when running from the research/ dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    print(f"[INFO] Current working directory: {os.getcwd()}")

    # Verify pred module exists relative to root
    target_file = os.path.join("src", "models", "predictor.py")
    print(f"[INFO] Checking file existence ({target_file}): {os.path.exists(target_file)}")

    try:
        from src.models.predictor import ProductionStockRegressor
        print("[SUCCESS] Import resolved ProductionStockRegressor successfully.")
        
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        
        # Fallback to inspect module namespace if class is missing
        try:
            import src.models.predictor as pred
            print(f"[DEBUG] Namespace attributes in predictor.py: {dir(pred)}")
        except Exception as load_err:
            print(f"[FATAL] Failed to load module entirely: {load_err}")

if __name__ == "__main__":
    main()