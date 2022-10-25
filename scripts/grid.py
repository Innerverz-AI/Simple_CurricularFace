import sys
sys.path.append("./")
import cv2
import glob
from PIL import Image
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageDraw, Image, ImageFont

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5]),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152

class CurrFace(nn.Module):
    def __init__(self, Backbone, InputSize, ckpt_path='./checkpoints/currface.pth'):
        super(CurrFace, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.currface = Backbone(InputSize).to(device)
        ckpt=torch.load(ckpt_path, map_location=device)
        self.currface.load_state_dict(ckpt)
        self.currface.eval()
        for param in self.currface.parameters():
            param.requires_grad = False
        del ckpt

    def forward(self, face):
        face = F.interpolate(face, (256,256)) #[36:108, 20:92, :] (256,352,768,864)
        # face_id = self.currface(F.interpolate(face[:, :, 88:216, 64:192], [128, 128], mode='bilinear', align_corners=True))
        face_id = self.currface(F.interpolate(face[:, :, 16:240, 16:240], [112, 112], mode='bilinear', align_corners=True))
        return face_id[0]

Backbone = IR_101
InputSize = [112, 112]

row_id = "KDWB"
col_id = "KDWB"

ckpt_path = "checkpoints/currface.pth"

currface = CurrFace(Backbone=Backbone, InputSize=InputSize, ckpt_path=ckpt_path)
row_img_paths = sorted(glob.glob(f"assets/{row_id}/*.*"))
col_img_paths = sorted(glob.glob(f"assets/{col_id}/*.*"))
dummy0 = np.zeros((256,256,3))
dummy1 = np.ones((256,256,3))*128
dummy2 = np.ones((256,256,3))*192
cs = nn.CosineSimilarity()

row_img_image_list = []
row_img_id_list = []
col_img_image_list = []
col_img_id_list = []
score_list = []

# row_img
for row_img_path in row_img_paths:
    row_img_image = Image.open(row_img_path).convert("RGB")
    row_img_tensor = transform(row_img_image).unsqueeze(0).cuda()
    # row_img_id_vector = currface(row_img_tensor.repeat(1,3,1,1))
    row_img_id_vector = currface(row_img_tensor)
    row_img_image_list.append(np.array(row_img_image.resize((256,256))))
    # row_img_image_list.append(np.array(row_img_image.resize((256,256)))[:, :, None].repeat(3, axis=2))
    row_img_id_list.append(row_img_id_vector)

# col_img
for col_img_path in col_img_paths:
    col_img_image = Image.open(col_img_path).convert("RGB")
    col_img_tensor = transform(col_img_image).unsqueeze(0).cuda()
    # col_img_id_vector = currface(col_img_tensor.repeat(1,3,1,1))
    col_img_id_vector = currface(col_img_tensor)
    col_img_image_list.append(np.array(col_img_image.resize((256,256))))
    # col_img_image_list.append(np.array(col_img_image.resize((256,256)))[:, :, None].repeat(3, axis=2))
    col_img_id_list.append(col_img_id_vector)

# score
score_list = []
for row_img_id_vector in row_img_id_list:
    row = []
    for col_img_id_vector in col_img_id_list:
        score = cs(row_img_id_vector, col_img_id_vector)
        row.append(np.round(float(score.detach().cpu().numpy()),2))
    score_list.append(row)
print(score_list)

# grid
grid = []

# first row
grid.append([dummy0]+row_img_image_list)

# second ~ last row
for i in range(len(col_img_paths)):
    row = [col_img_image_list[i]]
    for j in range(len(row_img_paths)):

        if j%2 == i%2:
            block = dummy1
        else: 
            block = dummy2
        
        block = Image.fromarray(block.astype(np.uint8))
        ImageDraw.Draw(block).text(
            xy=(32, 64),  # Coordinates
            text=str(score_list[j][i]),  # Text
            font=ImageFont.truetype("./assets/Arial.ttf", 80),
            fill=0,  # Color
        )
        block = np.array(block)

        row.append(block)
    grid.append(row)

grid = np.concatenate(grid, axis=1)
grid = np.concatenate(grid, axis=1)

cv2.imwrite(f"./assets/grids/grid_{row_id}_{col_id}.jpg", np.array(grid)[:, :, ::-1])
# cv2.imwrite(f"grid_{row_id}_{col_id}.jpg", np.array(grid))