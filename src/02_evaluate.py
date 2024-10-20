import torch
import yaml
import argparse
import json
import os
from models import DummyModelV3  # Import the model class

# Argument parsing to take the config file path
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()

# Load the configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Directly load the entire model (not just the state dictionary)
model = torch.load("models/model_C.pt")

# Print model version to confirm which model is being used
print(f"Evaluating with model version: {model.get_version()}")

# Simulate evaluation process (replace with actual evaluation code)
dummy_input = torch.randn(10, 10)  # Example input for evaluation
output = model(dummy_input)

# Calculate dummy metrics (replace with actual evaluation logic)
metrics = {
    "accuracy": 0.95,  # Dummy accuracy
    "f1_score": 0.93   # Dummy F1 score
}

# Ensure the metrics directory exists
os.makedirs("metrics", exist_ok=True)

# Save the evaluation metrics
with open("metrics/evaluation.json", "w") as f:
    json.dump(metrics, f)

print(f"Evaluation completed. Metrics saved to 'metrics/evaluation.json'.")
