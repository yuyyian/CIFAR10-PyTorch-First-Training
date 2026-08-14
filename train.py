from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 普通
    # log_dir = "./logs/CIFAR10/baseline"
    # 数据增强
    # log_dir = "./logs/CIFAR10/augmentation"
    # BatchNorm
    # log_dir = "./logs/CIFAR10/bn"
    # 数据增强 + BatchNorm
    log_dir = "./logs/CIFAR10/augmentation_bn"
    writer = SummaryWriter(log_dir)

    try:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor()
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=test_transform
        )

        train_datasize = len(train_dataset)
        test_datasize = len(test_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False
        )

        net = myNet().to(device)
        loss_fn = CrossEntropyLoss().to(device)

        learning_rate = 1e-2
        momentum = 0.9
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            momentum=momentum
        )

        epochs = 20
        best_epoch = 0
        best_accuracy = 0.0
        checkpoint_path = Path("checkpoints/best_model.pth")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            net.train()
            train_total_loss = 0.0
            train_correct = 0
            print(f"_______第 {epoch + 1} 轮训练开始______")

            for imgs, labels in train_dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output = net(imgs)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item() * imgs.size(0)
                train_correct += (output.argmax(1) == labels).sum().item()

            train_avg_loss = train_total_loss / train_datasize
            train_accuracy = train_correct / train_datasize
            writer.add_scalar("train_loss", train_avg_loss, epoch)
            writer.add_scalar("train_accuracy", train_accuracy, epoch)

            net.eval()
            test_total_loss = 0.0
            test_correct = 0
            with torch.no_grad():
                for imgs, labels in test_dataloader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    output = net(imgs)
                    loss = loss_fn(output, labels)

                    test_total_loss += loss.item() * imgs.size(0)
                    test_correct += (output.argmax(1) == labels).sum().item()

            test_avg_loss = test_total_loss / test_datasize
            test_accuracy = test_correct / test_datasize
            writer.add_scalar("test_loss", test_avg_loss, epoch)
            writer.add_scalar("test_accuracy", test_accuracy, epoch)

            print(
                f"train_loss: {train_avg_loss:.4f}, "
                f"train_accuracy: {train_accuracy:.2%}, "
                f"test_loss: {test_avg_loss:.4f}, "
                f"test_accuracy: {test_accuracy:.2%}"
            )

            if test_accuracy > best_accuracy:
                best_epoch = epoch + 1
                best_accuracy = test_accuracy
                torch.save(net.state_dict(), checkpoint_path)
                print(
                    f"保存最佳模型: epoch={best_epoch}, "
                    f"accuracy={best_accuracy:.2%}"
                )
            else:
                print("当前训练效果不如最佳模型")

        print(
            f"训练完成，最佳模型位于第 {best_epoch} 轮，"
            f"测试准确率为 {best_accuracy:.2%}"
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()

