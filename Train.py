import os

import torch
from tqdm import tqdm
from Model import Model
from Dataset import dataset_car
import Utils
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader


def train(model_fun, optim_fun, loss_fun, data_fun):
    data_iterator = tqdm(data_fun)

    for epoch in range(Utils.EPOCHS):
        for idx, (x, y) in enumerate(data_iterator):

            x = x.to(Utils.DEVICE, dtype=torch.float32).reshape(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
            y = y.to(Utils.DEVICE, dtype=torch.float32)

            pred = model_fun(x)

            loss_val = loss_fun(pred, y)

            print(loss_val)

            optim_fun.zero_grad()

            loss_val.backward()

            optim_fun.step()

            if idx % 1 == 0:

                if Utils.SAVE_MODEL:
                    with torch.no_grad():
                        path_save_model = Utils.SAVE_PATH + str(epoch) + "_" + str(idx) + ".pth"

                        with open(path_save_model, 'w') as fp:
                            pass

                        state = {
                            'epoch': idx,
                            'state_dict': model.state_dict(),
                            'optimizer': optim_fun.state_dict(),
                        }

                        torch.save(state, path_save_model)


model = Model(3, 10).to(device=Utils.DEVICE)
optim = torch.optim.Adam(params=model.parameters(), lr=0.001)
loss = torch.nn.L1Loss()
dataset = dataset_car(root="Data/train/", transform=transforms.ToTensor())
data = DataLoader(dataset=dataset, batch_size=Utils.BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    train(model, optim, loss, data)
