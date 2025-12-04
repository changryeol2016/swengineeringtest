import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======================
# 1. LeNet 정의 (CIFAR-10용)
# ======================
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 입력: 3x32x32
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)   # 3x32x32 -> 6x28x28
        self.pool = nn.MaxPool2d(2, 2)               # 6x28x28 -> 6x14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5) # 6x14x14 -> 16x10x10
        # pool -> 16x5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # CIFAR-10 클래스 10개

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 6, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 16, 5, 5]
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ======================
# 2. FGSM 공격 함수
# ======================
def fgsm_attack(model, criterion, images, labels, eps):
    # 입력에 대한 gradient를 구할 수 있도록 설정
    images.requires_grad = True

    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    # sign(∂loss/∂x)
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()

    # FGSM: x_adv = x + eps * sign(grad)
    perturbed_image = images + eps * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image

# ======================
# 3. 학습 및 평가 루프
# ======================
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.4f}")

def test_clean(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"[Clean] Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({acc:.2f}%)")
    return acc

def test_fgsm(model, device, test_loader, criterion, eps):
    model.eval()
    correct = 0
    adv_examples = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # FGSM 공격
        perturbed_data = fgsm_attack(model, criterion, data, target, eps)

        # 공격된 이미지로 다시 예측
        outputs = model(perturbed_data)
        pred = outputs.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print(f"[FGSM eps={eps}] Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")
    return acc

# ======================
# 4. 메인
# ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 데이터셋 & 전처리 (간단히 ToTensor만 사용)
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # 모델, 손실함수, 옵티마이저
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 간단히 5 epoch 정도만 학습 (원하시면 늘리시면 됩니다)
    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test_clean(model, device, test_loader, criterion)

    print("\n=== FGSM 공격 평가 ===")
    for eps in [0.0, 0.03, 0.1]:
        test_fgsm(model, device, test_loader, criterion, eps)

if __name__ == "__main__":
    main()

