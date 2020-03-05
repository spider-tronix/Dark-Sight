import torchvision.transforms as transforms

from results.configs import *

data_transform = transforms.Compose([  # TODO: Get the best of these
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

simple_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(SIZE),
    transforms.ToTensor(),
])

temps_normalize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Normalize(29.99, 1.049),
    transforms.ToTensor(),
])
