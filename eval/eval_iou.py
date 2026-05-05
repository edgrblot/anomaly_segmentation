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
        print(step)

        images = images.cuda()
        labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        logits = outputs.max(1)[1].unsqueeze(1).data.cuda()
        
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