#Imports

import os
import cv2

import torch
from torch import nn
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

# DataSet

class dataset_car(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root_dir = root
        self.transform = transform
        self.images = os.listdir(root);

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.root_dir + self.images[index]
        image = cv2.imread(img_path)
        x_image = image[:, :256, :]
        y_image = image[:, 256:, :]

        if self.transform:
            image = self.transform(x_image)
            y_image = self.transform(y_image)

        return x_image,y_image

# Model

class Model(nn.Module):
    def __init__(self, in_channels=3, num_blocks=6, features=None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(True),

            nn.Conv2d(features[0], features[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(True),

            nn.Conv2d(features[1], features[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(True),

            nn.Conv2d(features[2], features[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(True),
        )

        self.layers = []

        for i in range(num_blocks):
            self.layers.append(
                nn.Conv2d(features[3], features[3], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(features[3], features[2], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),

            nn.ConvTranspose2d(features[2], features[1], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),

            nn.ConvTranspose2d(features[1], features[0], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),

            nn.ConvTranspose2d(features[0], in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.down(x)
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.up(x)
        return x

# Utils

EPOCHS = 100
BATCH_SIZE = 1
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
SAVE_PATH = "Saved_Models/Model"


def train(model_fun, optim_fun, loss_fun, data_fun):
    data_iterator = tqdm(data_fun)

    for epoch in range(Utils.EPOCHS):
        for idx, (x, y) in enumerate(data_iterator):

            x = x.to(DEVICE, dtype=torch.float32).reshape(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
            y = y.to(DEVICE, dtype=torch.float32)

            pred = model_fun(x)

            loss_val = loss_fun(pred, y)

            print(loss_val)

            optim_fun.zero_grad()

            loss_val.backward()

            optim_fun.step()

            if idx % 1 == 0:

                if SAVE_MODEL:
                    with torch.no_grad():
                        path_save_model = SAVE_PATH + str(epoch) + "_" + str(idx) + ".pth"

                        with open(path_save_model, 'w') as fp:
                            pass

                        state = {
                            'epoch': idx,
                            'state_dict': model.state_dict(),
                            'optimizer': optim_fun.state_dict(),
                        }

                        torch.save(state, path_save_model)


model = Model(3, 10).to(device=DEVICE)
optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss = torch.nn.L1Loss()
dataset = dataset_car(root="Data/train/", transform=transforms.ToTensor())
data = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    train(model, optim, loss, data)