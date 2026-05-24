# Mask Architecture Anomaly Segmentation for Road Scenes
This is the repository of the work of Otmane Sarout (S350828) and Edgard Bouillot (S350869).
A comparison between mask and pixel based methods for road anomaly segmentation (using ERFNet and EoMT models, trained on the cityscapes dataset).
---

## Repository Structure

```
anomaly_segmentation/
├── eomt/               # EoMT model code
├── eval/               # ERFNet model code
├── trained_models/     
├── requirements.txt    # Python environnement dependencies
└── README.md
```

---

## Installation

The requirements for this project are to be found in the **requirements.txt** file.

```bash
# Clone the repo
git clone https://github.com/edgrblot/anomaly_segmentation
cd anomaly_segmentation

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
---

## Datasets

**Expected folder structure:**
```
ValidationDatasets/
├───cityscapes
│   ├───gtFine
│   │   ├───test
│   │   ├───train
│   │   └───val
│   └───leftImg8bit
│       ├───test
│       ├───train
│       └───val
├───FS_LostFound_full
│   ├───images
│   └───labels_masks
├───fs_static
│   ├───images
│   └───labels_masks
├───RoadAnomaly
│   ├───images
│   └───labels_masks
├───RoadAnomaly21
│   ├───images
│   └───labels_masks
└───RoadObsticle21
    ├───images
    └───labels_masks
```

---

## Usage
### mean Intersection over Union Evaluation 
To check mIoU for ERFNet model:
```bash
python .\eval\eval_miou.py
```

In order to evaluate the mIoU of the EoMT model: 
```bash
python .\eomt\miou_eval_launcher.py
```
### Post-hoc methods and temperature scaling
The evaluation of the ERFNet method is acheived running the `.\eval\evalAnomaly.py` script. The same script also implements the temperature scaling method.

```bash
python .\eval\evalAnomaly.py
```
The performances of the EoMT model are checked in the `.\eomt\inference.ipynb` notebook file.