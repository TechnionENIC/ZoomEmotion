from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import cv2

#Delete-vicki
from PIL import Image
import face_recognition
import time
import os
import random
import sys


# _, in_ch, num_categories, size_x, size_y, _ = get_dataset_params("children")

class MyLinear(nn.Linear):
    def __init__(self, in_feats, out_feats, drop_p, bias=True):
        super(MyLinear, self).__init__(in_feats, out_feats, bias=bias)
        self.masker = nn.Dropout(p=drop_p)

    def forward(self, x):
        masked_weight = self.masker(self.weight)
        return F.linear(x, masked_weight, self.bias)


class RonConv2D(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(RonConv2D, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

    def forward(self, x):
        return self.conv_block(x)


class RonNet(nn.Module):
    def __init__(self, in_ch, out_size, size_x, size_y):
        super(RonNet, self).__init__()
        input_to_linear = (size_x // 8) * (size_y // 8) * 256
        self.features = nn.Sequential(
            RonConv2D(in_ch, 64),
            RonConv2D(64, 128),
            RonConv2D(128, 256),
        )
        self.classifier = nn.Sequential(
            MyLinear(input_to_linear, 512, 0.25),
            nn.ReLU(),
            MyLinear(512, out_size, 0.5)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


CUDA_FLAG = False

dtype = torch.float32


class RonNetWrap(object):
    def __init__(self):
        self.size_x, self.size_y = 96, 96
        pth_file_to_use = "5 emotions"
        pth_files = {
            "5 emotions": "model_RonNet_balanced_fer5_greyscale.pth",
            "7 emotions": "model_RonNet_children_7_emotions_greyscale.pth",
            "7 emotions transfer": "model_RonNet_ck_greyscale_transfer_after_training.pth"
        }
        self.emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised',
                         'Neutral'] if pth_file_to_use in [
            "7 emotions", "7 emotions transfer"] else ['Angry', 'Fearful', 'Happy', 'Sad', 'Neutral']
        pth_file = pth_files[pth_file_to_use]
        in_ch = 1
        num_categories = 5
        self.net = RonNet(in_ch, num_categories, self.size_x, self.size_y)
        pth_dir = r"./"
        checkpoint = torch.load(pth_dir + pth_file, map_location=torch.device('cpu'))
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        #self.net.load_state_dict(torch.load(pth_dir + pth_file, map_location=torch.device('cpu')))

    # best img is of croped face
    def top_emotion(self, img):

        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.size_y, self.size_x))  # x and y are opposite since x is row, y is col
        #img = img / 255.0  # normalize
        # img11 = Image.fromarray(img, 'L')
        # curr_time = time.time()
        # img11.save(os.path.join(r'C:\Users\neuro\Desktop\Zoom_Emotion\all_photos',str(curr_time) + ".png"))
        img = torch.tensor(img, dtype=torch.float32)
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)
        if CUDA_FLAG:
            img = img.type(torch.cuda.FloatTensor)
            img = img.cuda()
        else:
            img = img.type(torch.FloatTensor)
        output = self.net(img)
        percentages = F.softmax(output, dim=1)[0]
        # noinspection PyArgumentList
        _, prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced

        eText, percentages = self.emotions[prediction.data.cpu().numpy()[0]], max(percentages.data.cpu().numpy())

        # file = open(os.path.join(r'C:\Users\neuro\Desktop\ZoomEmotion\all_photos',"emotion_ " + str(curr_time) + ".txt"), "w+")
        # file.write(" score: " + str(percentages) + " Emotion:" + eText)
        # file.close()
        return eText, percentages


if __name__ == "__main__":
    er = RonNetWrap()

    img = cv2.imread("crop_child.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # for gray images
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    cascPath = 'CascadeFiles/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)
    eText, percentages = er.top_emotion(gray)
    # percentages = np.round(percentages * 100, 2)
    print(eText, percentages)
