from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
path = Path(__file__).resolve().parents[2]

print(project_root)
print(path)