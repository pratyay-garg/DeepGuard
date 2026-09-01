with open('configs/quick_fusion.yaml', 'r') as f:
    content = f.read()

content = content.replace(
    "  audio:\\n    sample_rate: 16000",
    "  audio:\\n    return_raw_audio: true\\n    sample_rate: 16000"
)

with open('configs/quick_fusion.yaml', 'w') as f:
    f.write(content)
