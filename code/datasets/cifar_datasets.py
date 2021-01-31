import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pickle as pk
import yaml
import random


class LabeledCifar10(Dataset):
    def __init__(self, file_name, data_count=None, transforms=None):
        self.transforms = transforms
        with open(file_name, 'rb') as file:
            data = pk.load(file, encoding='latin1')

        if data_count is not None:
            self.images = []
            self.labels = []
            index = list(range(len(data['data'])))
            random.shuffle(index)
            choosen_index = index[: data_count]
            for i in choosen_index:
                self.images.append(data['data'][i])
                self.labels.append(data['labels'][i])
        else:
            self.images = data['data']
            self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transforms is not None:
            image = self.transforms(image)

        image = np.array((image.transpose(2, 0, 1) - 127.5) /
                         127.5, dtype=np.float32)

        return {'image': image, 'label': label}


class UnlabeledCifar10(Dataset):
    def __init__(self, file_name, transforms=None):
        self.tranforms = transforms
        with open(file_name, 'rb') as file:
            data = pk.load(file, encoding='latin1')

        self.images = data['data']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]

        if self.transforms is not None and self._run_transforms is True:
            image = self.transforms(image)

        image = np.array((image.transpose(2, 0, 1) - 127.5) /
                         127.5, dtype=np.float32)

        return {'image': image}


class PseudoLabeledCifar10(LabeledCifar10):
    def __init__(self, labeled_file_name, unlabeled_file_name, model, device, soft=True, transforms=None):
        super(PseudoLabeledCifar10, self).__init__(
            file_name=labeled_file_name, transforms=transforms)
        # Add pseudo labeled data
        with open(unlabeled_file_name, 'rb') as file:
            unlabeled_data = pk.load(file, encoding='latin1')

        if soft is True:
            for i, label in enumerate(self.labels):
                label_array = np.zeros(10, dtype=np.float32)
                label_array[label] = 1.0
                self.labels[i] = label_array

        model.to(device)
        model.eval()
        for image in unlabeled_data['data']:
            self.images.append(image)
            image = np.array((image.transpose(2, 0, 1) - 127.5) /
                             127.5, dtype=np.float32)
            with torch.no_grad():
                output = model(torch.from_numpy(image).unsqueeze(0).to(device))
            if soft is True:
                logit = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            else:
                logit = torch.max(output, dim=1)[1].item()

            self.labels.append(logit)


def cross_entropy_with_soft_target(pred, soft_targets):
    return torch.mean(torch.sum(- soft_targets + F.log_softmax(pred), 1))
