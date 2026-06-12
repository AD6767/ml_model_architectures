import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mini_resnet import MiniResNet


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_train_loader(batch_size: int = 64) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader

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


def main():
    device = get_device()
    print(f"Using device: {device}")
    train_loader = get_train_loader(batch_size=64)

    model = MiniResNet(num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    one_batch_overfit_test(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        steps=50
    )
    
if __name__ == '__main__':
    main()

