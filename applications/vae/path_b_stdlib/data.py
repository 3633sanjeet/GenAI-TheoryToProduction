import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def get_fashion_mnist_loaders(
    data_dir,
    batch_size=128,
    train_split=0.9,
    val_split=0.1,
    num_workers=0,
    download=True,
):
    """
    Load Fashion-MNIST dataset with train/val/test splits.

    Args:
        data_dir: Directory to store/load dataset
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation (rest is test)
        num_workers: Number of workers for data loading
        download: Whether to download dataset

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
    """

    # Keep in [0, 1] range from ToTensor (compatible with Sigmoid + BCE loss)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Alternative: simpler normalization to [0, 1]
    # transform = transforms.ToTensor()

    # Load full training set
    full_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )

    # Load test set
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )

    # Split training data into train/val
    n_train = len(full_train_dataset)
    n_val = int(n_train * val_split)
    n_train = n_train - n_val

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
        data_dir="../../data/",
        batch_size=4,
        num_workers=0
    )

    # Check shapes
    for batch in train_loader:
        images, labels = batch
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Min: {images.min():.4f}, Max: {images.max():.4f}")
        break

    print(f"✓ Data loading works!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
