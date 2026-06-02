# code running for the test sub folder (though shitty results are obtained), but not for val and train

import numpy as np
import torch
import os

from pathlib import Path
from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet import ERFNet
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

NUM_CLASSES = 20

# prediction class conversion
CITYSCAPES_ID_TO_TRAINID = {
    0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19,
    7: 0,  8: 1,  9: 19, 10: 19, 11: 2,  12: 3,  13: 4,
    14: 19, 15: 19, 16: 19, 17: 5,  18: 19, 19: 6,  20: 7,
    21: 8,  22: 9,  23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 19, 30: 19, 31: 16, 32: 17, 33: 18, -1: 19
}

# Get all the directories in ValidationDatasets
root_directory = Path(__file__).resolve().parents[1]

image_transform = ToPILImage()
input_transform_cityscapes = Compose([Resize(512, Image.BILINEAR), ToTensor()])
target_transform_cityscapes = Compose([Resize(512, Image.NEAREST), ToLabel(), Relabel(255, 19)])

def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
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

def remap_labels(labels_tensor):
    remapped = torch.full_like(labels_tensor, 19)
    for src, dst in CITYSCAPES_ID_TO_TRAINID.items():
        remapped[labels_tensor == src] = dst
    return remapped

def main(project_root):
    
    model = ERFNet(NUM_CLASSES)

    weightspath = os.path.join(project_root, "trained_models", "erfnet_pretrained.pth")
    dataset_directory = os.path.join(project_root, "ValidationDatasets", "cityscapes")

    model = torch.nn.DataParallel(model).cuda()
    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage, weights_only=True))
    model.eval()

    if(not os.path.exists(dataset_directory)):
        print ("Error: datadir could not be loaded")

    loader = DataLoader(cityscapes("ValidationDatasets", input_transform_cityscapes, target_transform_cityscapes, subset=r"\val"), num_workers=4, batch_size=2, shuffle=False)

    iouEvalVal = iouEval(NUM_CLASSES)

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        
        #print(step)

        images = images.cuda()
        labels = remap_labels(labels).cuda()
        
        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        logits = outputs.max(1)[1].unsqueeze(1).data.cuda()
        # After computing logits, clamp labels to valid range ([0, 19])
        labels = labels.clamp(0, NUM_CLASSES - 1)

        iouEvalVal.addBatch(logits, labels)

    iouVal, iou_classes = iouEvalVal.getIoU()
    iou_classes_str = []

    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    main(root_directory)