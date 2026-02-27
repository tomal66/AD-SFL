import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import datasets as hf_datasets

class HFWrapperDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, image_key="image", label_key="label", convert_mode=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.convert_mode = convert_mode

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item[self.image_key]
        
        if self.convert_mode and hasattr(image, "convert"):
            image = image.convert(self.convert_mode)
            
        label = item[self.label_key]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_datasets(dataset_name="MNIST", data_dir="./data", hf_token=None):
    """
    Downloads and returns the training and testing datasets from Hugging Face.
    """
    if dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = hf_datasets.load_dataset("ylecun/mnist", split="train", cache_dir=data_dir, token=hf_token)
        test_ds = hf_datasets.load_dataset("ylecun/mnist", split="test", cache_dir=data_dir, token=hf_token)
        train_dataset = HFWrapperDataset(train_ds, transform=transform, image_key="image", label_key="label", convert_mode="L")
        test_dataset = HFWrapperDataset(test_ds, transform=transform, image_key="image", label_key="label", convert_mode="L")
        
    elif dataset_name == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = hf_datasets.load_dataset("uoft-cs/cifar10", split="train", cache_dir=data_dir, token=hf_token)
        test_ds = hf_datasets.load_dataset("uoft-cs/cifar10", split="test", cache_dir=data_dir, token=hf_token)
        train_dataset = HFWrapperDataset(train_ds, transform=transform_train, image_key="img", label_key="label", convert_mode="RGB")
        test_dataset = HFWrapperDataset(test_ds, transform=transform_test, image_key="img", label_key="label", convert_mode="RGB")
        
    elif dataset_name == "CIFAR100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_ds = hf_datasets.load_dataset("uoft-cs/cifar100", split="train", cache_dir=data_dir, token=hf_token)
        test_ds = hf_datasets.load_dataset("uoft-cs/cifar100", split="test", cache_dir=data_dir, token=hf_token)
        train_dataset = HFWrapperDataset(train_ds, transform=transform_train, image_key="img", label_key="fine_label", convert_mode="RGB")
        test_dataset = HFWrapperDataset(test_ds, transform=transform_test, image_key="img", label_key="fine_label", convert_mode="RGB")
        
    elif dataset_name == "ImageNet":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_ds = hf_datasets.load_dataset("ILSVRC/imagenet-1k", split="train", cache_dir=data_dir, token=hf_token)
        test_ds = hf_datasets.load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=data_dir, token=hf_token)
        train_dataset = HFWrapperDataset(train_ds, transform=transform_train, image_key="image", label_key="label", convert_mode="RGB")
        test_dataset = HFWrapperDataset(test_ds, transform=transform_test, image_key="image", label_key="label", convert_mode="RGB")
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
        
    return train_dataset, test_dataset

def get_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
