import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from architectures.ch03_vit.tiny_vit import TinyViT


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_dataloaders(batch_size: int = 64, 
                    train_subset_size: int = 2000, 
                    test_subset_size: int = 500) -> tuple[DataLoader, DataLoader]:
    # 1. Transforms
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616),),
                                    ])
    # 2. Download data, get subset
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    train_subset = Subset(dataset=train_dataset, indices=range(train_subset_size))
    test_subset = Subset(dataset=test_dataset, indices=range(test_subset_size))
    # 3. Data loaders
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

def one_batch_overfit_test(model: nn.Module,
                           train_loader: DataLoader,
                           criterion: nn.Module,
                           optimizer: optim.Optimizer,
                           device: torch.device,
                           steps: int = 10) -> None:
    model.train()

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device) # [B,]

    for step in range(steps):
        optimizer.zero_grad()

        logits = model(images) #[B, num_classes]
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1) # [B,]
        accuracy = (1.0 * (preds == labels).sum()) / preds.shape[0]

        if step % 10 == 0 or step == steps - 1:
            print(f"step: {step} | loss: {loss.item():.4f} | acc: {accuracy * 100:.2f}")

def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device) # (B,)

        optimizer.zero_grad()

        logits = model(images) # (B, num_classes)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        preds = torch.argmax(logits, dim=1) # (B,)
        correct = (preds == labels).sum().item()
        total_correct += correct
        
        total_examples += batch_size
        total_loss += loss.item() * batch_size

    avg_accuracy = total_correct / total_examples
    avg_loss = total_loss / total_examples
    return avg_accuracy, avg_loss

def evaluate(model: nn.Module,
             test_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device) # (B,)

            logits = model(images) # (B, num_classes)
            loss = criterion(logits, labels)

            batch_size = labels.shape[0]
            preds = torch.argmax(logits, dim=1) # (B,)
            correct = (preds == labels).sum().item()
            total_correct += correct

            total_examples += batch_size
            total_loss += loss.item() * batch_size

    avg_accuracy = total_correct / total_examples
    avg_loss = total_loss / total_examples
    return avg_accuracy, avg_loss


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(
        batch_size=64,
        train_subset_size=2000,
        test_subset_size=500,
    )

    model = TinyViT(
        image_size=32,
        patch_size=4,
        channels=3,
        embed_dim=64,
        num_heads=4,
        num_blocks=2,
        mlp_ratio=4,
        num_classes=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-4)

    # one_batch_overfit_test(
    #     model=model,
    #     train_loader=train_loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     device=device,
    #     steps=100,
    # )
    num_epochs = 3
    for epoch in range(num_epochs):
        train_acc, train_loss= train_one_epoch(model=model,
                                                train_loader=train_loader,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                device=device)
        test_acc, test_loss = evaluate(model=model,
                                       test_loader=test_loader,
                                       criterion=criterion,
                                       device=device)
        print(
        f"epoch {epoch + 1}/{num_epochs} | "
        f"train loss: {train_loss:.4f} | "
        f"train acc: {train_acc * 100:.2f}% | "
        f"test loss: {test_loss:.4f} | "
        f"test acc: {test_acc * 100:.2f}%"
    )
        

if __name__ == '__main__':
    main()
