import torch
import yaml
import argparse
import os
from models import DummyModelV2  # Ensure the model class is imported from models.py

# Argument parsing for config
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()

# Load the configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Instantiate and train the model
model = DummyModelV2()

# Print model version to confirm which model is being used
print(f"Training {model.get_version()}")

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# Save only the model's state dictionary (weights)
torch.save(model.state_dict(), "models/model_v2.pt")

# Save the preprocessing script in the src/ directory
with open("src/preprocess.py", "w") as f:
    f.write("""
def preprocess_audio_data(input_path):
    # Dummy preprocessing logic for the model
    return [0] * 16000  # Dummy data simulating audio input
""")

# Save the post-processing script in the src/ directory
with open("src/postprocess.py", "w") as f:
    f.write("""
def postprocess_anomaly_detection(output, config):
    # Dummy post-processing logic
    return {"anomaly_score": output.item(), "is_anomalous": output.item() > config['threshold']}
""")

print(f"{model.get_version()} training completed. Model state saved as 'models/model_v2.pt'.")
print("Preprocessing and post-processing scripts saved.")
