#!/usr/bin/env python
"""
Script simple pour lancer la validation
"""

import subprocess
import os
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent.parent
# Chemin du projet
#path = PROJECT_ROOT / "trained_models"

#project_root = r"\courses\data_analysis_and_artificial_intelligence\project"
"""eomt_dir = os.path.join(project_root, "MaskArchitectureAnomaly_CourseProject", "eomt")

print("=" * 80)
print("Lancement de la validation avec RbA")
print("=" * 80)
print(f"\nRépertoire EOMT: {eomt_dir}")
print(f"Python: {python_exe}\n")
"""
python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")
print(python_exe)
# Changer de répertoire
#os.chdir(eomt_dir)

# Configurer l'environnement
os.environ["WANDB_MODE"] = "offline"

# Commande
cmd = [
    python_exe,
    "eomt/main.py",
    "validate",
    "-c", "eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
    "--data.path", "./ValidationDatasets",
    "--model.ckpt_path", "eomt/eomt_cityscapes.bin",
    "--trainer.max_epochs", "1",
    "--data.batch_size", "2"
]

print("Commande:")
print(" ".join(cmd))
print("\nExécution...\n")
print("=" * 80)

# Exécuter
result = subprocess.run(cmd)

print("=" * 80)
print(f"\nExit code: {result.returncode}")

sys.exit(result.returncode)