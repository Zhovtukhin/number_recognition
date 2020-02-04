import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torchvision import transforms

from mean_std import mean_std


# Determine the model architecture
class LeNetBN(nn.Module):
    def __init__(self):
        super().__init__()

        # convolution layers
        self._body = nn.Sequential(
            # First convolution Layer
            # input size = (32, 32), output size = (28, 28)
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(6),
            # ReLU activation
            nn.ReLU(inplace=True),
            # Max pool 2-d
            nn.MaxPool2d(kernel_size=2),

            # Second convolution layer
            # input size = (14, 14), output size = (10, 10)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # output size = (5, 5)
        )

        # Fully connected layers
        self._head = nn.Sequential(
            # First fully connected layer
            # in_features = total number of weight in last conv layer = 16 * 5 * 5
            nn.Linear(in_features=16 * 5 * 5, out_features=120),

            # ReLU activation
            nn.ReLU(inplace=True),

            # second fully connected layer
            # in_features = output of last linear layer = 120
            nn.Linear(in_features=120, out_features=84),

            # ReLU activation
            nn.ReLU(inplace=True),

            # Third fully connected layer. It is also output layer
            # in_features = output of last linear layer = 84
            # and out_features = number of classes = 10 (data 0-9)
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        # apply feature extractor
        x = self._body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weight_in_last conv_layer
        x = x.view(x.size()[0], -1)
        # apply classification head
        x = self._head(x)
        return x


if __name__ == '__main__':

    image = np.array([])
    model_path = "models/MNIST_SVHM_aug_model.pth"
    mean, std = mean_std["MNIST_SVHM_aug_model"]

    if len(sys.argv) <= 1:
        sys.exit('Need image PATH')
    elif len(sys.argv) <= 4:
        try:
            image = cv2.imread(sys.argv[1], 0)
        except:
            sys.exit('Image file do not exist')
    elif len(sys.argv) > 4:
        try:
            image = cv2.imread(sys.argv[1], 0)
        except:
            sys.exit('Image file do not exist')
        model_path = sys.argv[2]
        try:
            mean, std = int(sys.argv[3]), int(sys.argv[4])
        except:
            sys.exit('Mean and std must be Integers')

    if image is None:
        sys.exit('Image file do not exist')

    image = cv2.copyMakeBorder(image, 14, 14, 14, 14, cv2.BORDER_CONSTANT, None, 0)

    # Find connected components
    etval, labels, stats, centroid = \
        cv2.connectedComponentsWithStatsWithAlgorithm(image, 4, cv2.CV_16U, cv2.CCL_WU)

    # If find less then 4 digits - print that
    nComponents = labels.max()
    #if nComponents < 4:
        #print("Numbers in " + sys.argv[1] + " are too close, program will not find all digits")

    model_param = []  # for plotting result of testing models

    # Download model
    model = LeNetBN()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except:
        sys.exit('Model file do not exist')

    number = []
    for i in range(1, nComponents + 1):
        # Crop image (28, 28) with one digit in the center
        centerX, centerY = map(int, centroid[i])
        img = np.where(labels == i, image, 0)
        img = img[centerY - 14:centerY + 14, centerX - 14:centerX + 14]
        # Prepare for model input
        img = torch.tensor(img.astype(np.uint8))
        img = img[None]
        img = transforms.ToPILImage()(img)
        img = transforms.Resize((32, 32))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((mean,), (std,))(img)
        img = img[None]
        # Predict
        score = model(img)
        prob = nn.functional.softmax(score[0], dim=0)
        y_pred = prob.argmax()
        number.append([centroid[i][0], y_pred.cpu().data.numpy().tolist()])

    # Sort by X coordinate
    number.sort()
    number = ''.join([str(i[1]) for i in number])
    print(number)
