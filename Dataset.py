from torch.utils.data import Dataset
import os
import cv2


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
