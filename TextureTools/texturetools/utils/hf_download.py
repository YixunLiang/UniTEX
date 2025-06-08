# export HF_HOME="/home/chenxiao/huggingface"
# export HF_ENDPOINT="https://hf-mirror.com"
import huggingface_hub
for rep in [
    'XLabs-AI/flux-lora-collection',
]:
    huggingface_hub.snapshot_download(rep)
