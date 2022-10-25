import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import time
import numpy as np
import cv2
from PIL import Image
from backbone.model_irse import IR_101
from torchvision import transforms


root = "./assets/JIN"

BACKBONE = IR_101([112, 112])    
BACKBONE.cuda()
BACKBONE.eval()
BACKBONE.load_state_dict(torch.load(f"checkpoints/checkpoint.pt"))

TC = transforms.Compose([
    transforms.CenterCrop(192),
    transforms.Resize(112),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


img1 = Image.open("assets/JIN/jin1.jpg").convert("RGB")
img2 = Image.open("assets/JIN/jin2.jpg").convert("RGB")

img1 = TC(img1).unsqueeze(0).cuda()
img2 = TC(img2).unsqueeze(0).cuda()

vec1 = BACKBONE(img1)
vec2 = BACKBONE(img2)

cs = nn.CosineSimilarity()
score = cs(vec1[0], vec2[0])
print(score)