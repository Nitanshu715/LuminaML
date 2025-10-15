import os
import sys

# Automatically add the src folder to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)
from src.train import train_model
from src.predict import predict

if __name__ == "__main__":
    print("🚀 Starting Amazon ML pipeline...")
    train_model()
    predict()
    print("✅ All done! Output saved in student_resource/submission/test_out.csv")
