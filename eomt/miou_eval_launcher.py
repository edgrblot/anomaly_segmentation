import subprocess
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")

# setting wandb as offline
os.environ["WANDB_MODE"] = "offline"

# Command
cmd = [
    python_exe,
    "eomt/main.py",
    "validate",
    "-c", "eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
    "--data.path", r"C:\Users\edgar\Documents\segmentation_project\code\anomaly_segmentation\ValidationDatasets",
    "--model.ckpt_path", "eomt/eomt_cityscapes.bin",
    "--trainer.max_epochs", "1",
    "--data.batch_size", "2"
]

# run the code
result = subprocess.run(cmd)
sys.exit(result.returncode)