
def postprocess_anomaly_detection(output, config):
    # Dummy post-processing logic
    return {"anomaly_score": output.item(), "is_anomalous": output.item() > config['threshold']}
