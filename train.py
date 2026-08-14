import argparse
from pathlib import Path

import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from models.cnn import myNet


EXPERIMENTS = {
    "baseline": {"use_augmentation": False, "use_bn": False},
    "augmentation": {"use_augmentation": True, "use_bn": False},
    "bn": {"use_augmentation": False, "use_bn": True},
    "augmentation_bn": {"use_augmentation": True, "use_bn": True},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练 CIFAR-10 CNN 对照实验"
    )
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS,
        required=True,
        help="选择要运行的实验配置",
    )
    return parser.parse_args()


def build_train_transform(use_augmentation):
    transforms = []
    if use_augmentation:
        transforms.extend([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ])
    transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(transforms)


def main():
    args = parse_args()
    experiment = EXPERIMENTS[args.experiment]
    use_augmentation = experiment["use_augmentation"]
    use_bn = experiment["use_bn"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(
        f"实验配置: {args.experiment} "
        f"(data augmentation={use_augmentation}, batch norm={use_bn})"
    )

    log_dir = Path("logs/CIFAR10") / args.experiment
    writer = SummaryWriter(str(log_dir))

    try:
        train_transform = build_train_transform(use_augmentation)
        test_transform = torchvision.transforms.ToTensor()

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

        net = myNet(use_bn=use_bn).to(device)
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
        checkpoint_path = (
            Path("checkpoints") / args.experiment / "best_model.pth"
        )
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

