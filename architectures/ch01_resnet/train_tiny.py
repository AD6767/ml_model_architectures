import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mini_resnet import MiniResNet


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_dataloaders(batch_size: int = 64, 
                     train_subset_size: int = 2000, 
                     test_subset_size: int = 500) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_subset = Subset(dataset=train_dataset, indices=range(train_subset_size))
    test_subset = Subset(dataset=test_dataset, indices=range(test_subset_size))

    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def one_batch_overfit_test(model: nn.Module, 
                           train_loader: DataLoader, 
                           criterion: nn.Module, 
                           optimizer: optim.Optimizer,
                           device: torch.device,
                           steps: int = 5) -> None:
    model.train()

    images, labels = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    for step in range(steps):
        # 1. zero gradients -- clear old gradients
        optimizer.zero_grad()
        # 2. forward pass
        logits = model(images)
        # 3. compute loss
        loss = criterion(logits, labels)
        # 4. backward pass -- compute gradients
        loss.backward()
        # 5. optimizer step -- update weights
        optimizer.step()
        # 6. compute batch accuracy
        preds = torch.argmax(logits, dim=1) # B,num_classes -> (B,1)
        correct = (preds == labels).sum().item() # item(): Strips away the PyTorch tensor container and returns a pure Python integer
                                                 # sum(): Adds them all up. In PyTorch, True counts as 1 and False counts as 0.
        total = labels.size(0)
        accuracy = correct / total

        # 7. print every 10 steps
        if step % 10 == 0 or step == steps - 1:
            print(f"step: {step:03d} | loss: {loss.item():.4f} | acc: {accuracy * 100: .2f}")

def train_one_epoch(model: nn.Module, 
                    train_loader: DataLoader, 
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += batch_size

    avg_loss = total_loss / total_examples
    avg_correct = total_correct / total_examples
    return avg_loss, avg_correct

def evaluate(model: nn.Module, 
            data_loader: DataLoader, 
            criterion: nn.Module,
            device: torch.device) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_examples += batch_size

    avg_loss = total_loss / total_examples
    avg_correct = total_correct / total_examples
    return avg_loss, avg_correct


def main():
    device = get_device()
    print(f"Using device: {device}")
    train_loader, test_loader = get_dataloaders(
        batch_size=64,
        train_subset_size=2000,
        test_subset_size=500,
    )

    model = MiniResNet(num_classes=10)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    # Optional sanity check:
    # one_batch_overfit_test(
    #     model=model,
    #     train_loader=train_loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     device=device,
    #     steps=50
    # )

    num_epochs = 3
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = evaluate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"epoch {epoch + 1:02d}/{num_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc * 100:.2f}% | "
            f"test loss: {test_loss:.4f} | "
            f"test acc: {test_acc * 100:.2f}%"
        )
    
if __name__ == '__main__':
    main()

