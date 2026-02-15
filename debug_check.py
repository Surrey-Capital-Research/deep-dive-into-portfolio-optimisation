import os
import sys

# Print current directory
print(f"Current Path: {os.getcwd()}")

# Check if the file exists
target_file = os.path.join("src", "models", "predictor.py")
print(f"Checking for {target_file}: {os.path.exists(target_file)}")

# Try to force the import and catch the specific error
try:
    from src.models.predictor import ProductionStockRegressor
    print("✅ Success! Import found the class.")
except Exception as e:
    print(f"❌ Failed: {e}")
    # Show what IS inside the module if it found the file
    try:
        import src.models.predictor as pred
        print(f"Attributes found in predictor.py: {dir(pred)}")
    except:
        print("Could not even load the file 'predictor.py'")