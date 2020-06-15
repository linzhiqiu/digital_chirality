import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader, DatasetFolder
from PIL import Image
from utils import mosaic, demosaic, rand_rgb_image
import cv2
import numpy as np

def get_transform(image_size, crop, crop_size):
    assert crop in ["none", 'random_crop_inside_boundary']

    transforms_list_train = []
    transforms_list_test = []

    if crop == "none":
        print("No cropping to the images.")
    elif crop == "random_crop_inside_boundary":
        boundary = image_size - 32 # 16 pixels boundary
        print(f"First performing a center crop of size {boundary} to avoid boundary")
        transforms_list_train += [
            transforms.CenterCrop(boundary),
            transforms.RandomCrop(crop_size)
        ]
        transforms_list_test += [
            transforms.CenterCrop(boundary),
            transforms.RandomCrop(crop_size)
        ]

    transforms_list_train += [transforms.ToTensor()]
    transforms_list_test += [transforms.ToTensor()]

    data_transforms = {
        'train': transforms.Compose(transforms_list_train),
        'test': transforms.Compose(transforms_list_test),
    }
    return data_transforms

def get_dataloaders(train_size=100000,
                    val_size=5000,
                    image_pattern='gaussian_rgb',
                    demosaic_algo='Malvar2004',
                    bayer_pattern='RGGB',
                    jpeg_coeff=25,
                    image_size=576,
                    image_type='original',
                    crop='random_crop_inside_boundary',
                    crop_size=512,
                    batch_size=4,
                    num_workers=4):
    '''
        Return a factory of PyTorch dataset/dataloader
    '''
    data_transform = get_transform(image_size, crop, crop_size) # A dict with 'train' 'test'

    train_dataset = ChiralDataset(train_size,
                                  data_transform['train'],
                                  image_type=image_type,
                                  image_size=image_size,
                                  image_pattern=image_pattern,
                                  demosaic_algo=demosaic_algo,
                                  bayer_pattern=bayer_pattern,
                                  jpeg_coeff=jpeg_coeff)
    val_dataset   = ChiralDataset(val_size,
                                  data_transform['test'],
                                  image_type=image_type,
                                  image_size=image_size,
                                  image_pattern=image_pattern,
                                  demosaic_algo=demosaic_algo,
                                  bayer_pattern=bayer_pattern,
                                  jpeg_coeff=jpeg_coeff,)

    return {
        'train' : torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False,
                                              num_workers=num_workers,
                                              pin_memory=False),
        'val' : torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=False),
    }

class ChiralDataset(torch.utils.data.Dataset):
    def __init__(self,
                 size,
                 transform,
                 image_size,
                 image_type,
                 image_pattern,
                 demosaic_algo,
                 bayer_pattern,
                 jpeg_coeff=25):
        """A dataset that contains randomly generated images with random flip
            Args:
            size - the size of dataset
            transform - the transformation to be applied to images (e.g. random crop)
            image_size - The size of image
            image_type - the type of image (original/demosaic/jpeg/both)
            image_pattern - the distribution of images
            demosaic_algo - If the image undergone demosaicing step, then use this demosaic algorithm
            bayer_pattern - If the image undergone demosaicing step, then use this bayer grid pattern
            jpeg_coeff - The jpeg compression coefficient if using JPEG-based image_type.
        """
        self.class_names = ["flipped", "original"]
        self.size = size

        self.horizontalFlip = torchvision.transforms.RandomHorizontalFlip(p=1)
        self.transform = transform
        self.image_type = image_type
        self.image_size = image_size
        self.image_pattern = image_pattern
        self.demosaic_algo = demosaic_algo
        self.bayer_pattern = bayer_pattern

        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_coeff]

        self.shared_batch_base_seed = 0 # Should be the epoch number

    def __getitem__(self, index):
        np.random.seed(self.shared_batch_base_seed * self.size + int(index/2))
        image_original = rand_rgb_image(self.image_size, self.image_pattern)

        if self.image_type == 'original':
            img = image_original
        elif self.image_type == 'jpeg':
            img = cv2.cvtColor(image_original, cv2.COLOR_RGB2BGR)
            _, img = cv2.imencode('.jpg', img, self.encode_param)
            img = cv2.imdecode(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            image_demosaiced = demosaic(
                mosaic(image_original, pattern=self.bayer_pattern),
                pattern=self.bayer_pattern,
                algo=self.demosaic_algo
            ).astype('uint8')
            if self.image_type == 'demosaic':
                img = image_demosaiced
            elif self.image_type == 'both':
                both_new = cv2.cvtColor(image_demosaiced, cv2.COLOR_RGB2BGR)
                _, both_new = cv2.imencode('.jpg', both_new, self.encode_param)
                both_new = cv2.imdecode(both_new, 1)
                img = cv2.cvtColor(both_new, cv2.COLOR_BGR2RGB)
        sample = Image.fromarray(img)
        
        if index % 2 == 0:
            sample = self.horizontalFlip(sample)
            label = 0 # Flip is 0
        else:
            label = 1 # Original is 1

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return self.size

