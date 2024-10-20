import yaml
import argparse
import torch
import os
import json
from models import DummyModelV3  # Import the new model class

# Argument parsing for config
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()

# Load the configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Load the trained model
model = torch.load("models/model_v3.pt")  # Load the new model

# Print model version to confirm which model is being evaluated
print(f"Evaluating {model.get_version()}")

# Simulate evaluation (in practice, you'd evaluate the model on validation/test data)
metrics = {
    "accuracy": 0.97,  # Updated dummy metric for the new model
    "f1_score": 0.93   # Updated dummy metric for the new model
}

# Ensure the metrics directory exists
os.makedirs("metrics", exist_ok=True)

# Save the evaluation metrics
with open("metrics/evaluation.json", "w") as f:
    json.dump(metrics, f)

print(f"Evaluation completed. Metrics saved to 'metrics/evaluation.json'.")
