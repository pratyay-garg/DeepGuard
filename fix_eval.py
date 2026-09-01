with open('scripts/evaluate.py', 'r') as f:
    content = f.read()

old_model_seq = """    # Create model using factory
    config["model"]["sequence_length"] = config.get("data", {}).get("sequence_length", 16)
    model = create_model("""
new_model_seq = """    # Create model using factory
    if "temporal" in config.get("model", {}):
        config["model"]["sequence_length"] = config["model"]["temporal"].get("sequence_length", 16)
    else:
        config["model"]["sequence_length"] = config.get("data", {}).get("sequence_length", 16)
    model = create_model("""
content = content.replace(old_model_seq, new_model_seq)

with open('scripts/evaluate.py', 'w') as f:
    f.write(content)
