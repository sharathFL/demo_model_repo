import torch
import yaml
import argparse
import os
from models import DummyModelV2  # Import the model class

# Argument parsing for config
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()

# Load the configuration file
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

# Instantiate the model (the same model class you used in training)
model = DummyModelV2()

# Load the saved state dictionary (weights)
model.load_state_dict(torch.load("models/model_v2.pt"))

# Now you can access model methods like `get_version()`
print(f"Evaluating {model.get_version()}")

# Simulate evaluation (you can replace this with actual evaluation code)
dummy_input = torch.tensor([0.5] * 10)  # Dummy input
output = model(dummy_input)

# Example evaluation metrics (replace this with real ones)
metrics = {
    "accuracy": 0.95,  # Dummy accuracy
    "f1_score": 0.93   # Dummy F1 score
}

# Ensure the metrics directory exists
os.makedirs("metrics", exist_ok=True)

# Save the evaluation metrics
with open("metrics/evaluation.json", "w") as f:
    yaml.dump(metrics, f)

print(f"Evaluation completed. Metrics saved to 'metrics/evaluation.json'.")
