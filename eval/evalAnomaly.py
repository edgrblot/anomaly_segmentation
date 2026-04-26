import os
import glob
import torch
import random
import numpy as np

from PIL import Image
from erfnet import ERFNet
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
from pathlib import Path

# Configuration to ensure reproducibility of results and model parameters.
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# GPU training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Transformations for input images (resizing + tensor conversion) and for ground truth masks (resizing without interpolation)
input_transform = Compose([Resize((512, 1024), Image.BILINEAR),ToTensor()])
target_transform = Compose([Resize((512, 1024), Image.NEAREST)])

# Defining the number of classes of the trained model
NUM_CLASSES = 20

# Get all the directories in ValidationDatasets
root_directory = Path(__file__).resolve().parents[1]
datasets_directory = "ValidationDatasets"
images_directory = "images"
labels_directory = "labels_masks"
dataset_directory = os.path.join(root_directory, datasets_directory)

# The list of all the datasets in dataset_directory
directories = [entry.name for entry in os.scandir(dataset_directory) if entry.is_dir()]
# training_datasets = [os.path.join(dataset_directory, directory) for directory in directories]
training_datasets = [os.path.join(dataset_directory, "fs_static")]

def calculate_msp(logits):
    """
    Calculates Maximum Softmax Probability, using the logits. Exact definition as the one given by the teacher.
    """
    return 1.0 - np.max(logits, axis=0)

def calculate_entropy(logits):
    """
    Calculates the entropy from the logits. Definition established from the one defined in the SegmentMeIfYouCan project.
    """
    softmax_output = np.exp(logits) / np.sum(np.exp(logits), axis=0, keepdims=True)
    log_probs = np.log(softmax_output)# + 1e-10)
    return np.sum(-softmax_output * log_probs, axis=0)

def calculate_max_logit(logits):
    """
    Calculates the maximum logit. Definition from Scaling OoD Detection for Real-World Settings
    """
    return -np.max(logits, axis=0)

def calculate_metrics(scores, ood_mask, ind_mask):
    """
    Calculates AUPRC and FPR@TPR95, using the initial implemented method.
    """
    ood_out = scores[ood_mask]
    ind_out = scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    return prc_auc, fpr

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                print(name, " not loaded")
                continue
        else:
            own_state[name].copy_(param)
    return model

def erfnet_perf_eval(training_datasets, num_classes, project_root):
    
    MainPath = Path(__file__).resolve().parents[1] / "trained_models"
    loadDir = str(MainPath)
    loadWeights = "/erfnet_pretrained.pth"

    if not os.path.exists(r'.\eval\results.txt'):
        open(r'.\eval\results.txt', 'w').close()
    file = open(r'.\eval\results.txt', 'a')
    
    weightspath = loadDir + loadWeights

    # defining which model to use
    model = ERFNet(NUM_CLASSES)

    model = torch.nn.DataParallel(model).cuda()
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage, weights_only=True))
    
    model.eval()

    # Loop through each dataset for evaluation
    for dataset_dir in training_datasets:

        dataset_name = os.path.basename(dataset_dir)
        images_pattern = os.path.join(dataset_dir, images_directory, "*.*").replace('/', '\\')

        msp_scores = []
        entropy_scores = []
        maxlogit_scores = []
        ood_gts_list = []
        
        # list of all the images in the directory
        image_paths = glob.glob(os.path.abspath(images_pattern))
        
        # each "path" is the full path to an image of the data set
        # so loop over the images of the dataset
        for path in image_paths:
            
            # Obtain image extension
            image_ext = os.path.splitext(path)[1].lower()

            # Determine corresponding ground truth label path
            pathGT = path.replace(images_directory, labels_directory)
            if image_ext == '.webp':
                pathGT = pathGT.replace('.webp', '.png')
           
            elif image_ext == '.jpg':
                pathGT = pathGT.replace('.jpg', '.png')

            if not os.path.exists(pathGT):
                print(f"The {pathGT} label doesn't exist.")
                continue

            image = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
            
            with torch.no_grad():
                result = model(image)

            # Obtain predicted class labels
            predictions = torch.argmax(result, dim=1).squeeze().cpu().numpy()
            mask = Image.open(pathGT)
            mask = target_transform(mask)
            ood_gts = np.array(mask)

            logits = result.squeeze(0).data.cpu().numpy()

            # calculate anomaly scores
            msp_result = calculate_msp(logits)
            entropy_result = calculate_entropy(logits)
            maxlogit_result = calculate_max_logit(logits)

            if "RoadAnomaly" in dataset_name:
                ood_gts = np.where((ood_gts==2), 1, ood_gts)

            elif "LostAndFound" in dataset_name:
                ood_gts = np.where((ood_gts==0), 255, ood_gts)
                ood_gts = np.where((ood_gts==1), 0, ood_gts)
                ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

            if 1 not in np.unique(ood_gts):
                print(f"No anomaly in {pathGT[-6:]}")
                continue

            else:
                ood_gts_list.append(ood_gts)
                msp_scores.append(msp_result)
                entropy_scores.append(entropy_result)
                maxlogit_scores.append(maxlogit_result)

            torch.cuda.empty_cache()

        if len(ood_gts_list) == 0 or len(msp_scores) == 0:
            print(f"No data collected for the dataset {dataset_name}. Check the paths of images and labels.")
            continue

        msp_scores = np.array(msp_scores)
        entropy_scores = np.array(entropy_scores)
        maxlogit_scores = np.array(maxlogit_scores)

        ood_gts = np.array(ood_gts_list)

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)
        
        # Calculate the metrics for each method
        msp_auc, msp_fpr = calculate_metrics(msp_scores, ood_mask, ind_mask)
        entropy_auc, entropy_fpr = calculate_metrics(entropy_scores, ood_mask, ind_mask)
        maxlogit_auc, maxlogit_fpr = calculate_metrics(maxlogit_scores, ood_mask, ind_mask)

        # Display and write the results
        if msp_auc is not None:
            print(f'MSP         AUPRC       {msp_auc*100.0}')
            print(f'MSP         FPR@TPR95   {msp_fpr*100.0}')
            file.write(f'Dataset: {dataset_name}\n')
            file.write(f'AUPRC score (MSP): {msp_auc*100.0}\n')
            file.write(f'FPR@TPR95 (MSP): {msp_fpr*100.0}\n')

        if entropy_auc is not None:
            print(f'Entropy     AUPRC       {entropy_auc*100.0}')
            print(f'Entropy     FPR@TPR95   {entropy_fpr*100.0}')
            file.write(f'AUPRC score (Entropy): {entropy_auc*100.0}\n')
            file.write(f'FPR@TPR95 (Entropy): {entropy_fpr*100.0}\n')

        if maxlogit_auc is not None:
            print(f'MaxLogit    AUPRC       {maxlogit_auc*100.0}')
            print(f'MaxLogit    FPR@TPR95   {maxlogit_fpr*100.0}')
            file.write(f'AUPRC score (MaxLogit): {maxlogit_auc*100.0}\n')
            file.write(f'FPR@TPR95 (MaxLogit): {maxlogit_fpr*100.0}\n')

        file.write('\n')

    file.close()

if __name__ == '__main__':
    erfnet_perf_eval(training_datasets, NUM_CLASSES, root_directory)