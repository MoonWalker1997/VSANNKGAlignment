from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

# Data preprocessing, transform to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
# Batch size is predefined
train_dataset = datasets.MNIST(root="./Data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = datasets.MNIST(root="./Data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
