import argparse
from pathlib import Path
import torch
import torchvision
from torch.nn import CrossEntropyLoss
from models.cnn import myNet


EXPERIMENTS_WITH_BN = {"bn", "augmentation_bn"}
EXPERIMENTS = ("baseline", "augmentation", "bn", "augmentation_bn")


def parse_args():
    parser = argparse.ArgumentParser(
        description="测试 CIFAR-10 CNN 对照实验的最佳模型"
    )
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENTS,
        required=True,
        help="选择要测试的实验配置",
    )
    return parser.parse_args()


def resolve_checkpoint_path(experiment):
    checkpoint_path = Path("checkpoints") / experiment / "best_model.pth"
    if checkpoint_path.exists():
        return checkpoint_path

    raise FileNotFoundError(
        f"未找到 {checkpoint_path}，请先运行 "
        f"python train.py --experiment {experiment}"
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(args.experiment)

    test_dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    model = myNet(use_bn=args.experiment in EXPERIMENTS_WITH_BN).to(device)
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    loss_fn = CrossEntropyLoss().to(device)

    model.eval()
    test_total_loss = 0.0
    test_correct = 0
    with torch.no_grad():
        for imgs, labels in test_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs)
            loss = loss_fn(output, labels)
            test_total_loss += loss.item() * imgs.size(0)
            test_correct += (output.argmax(1) == labels).sum().item()

    test_avg_loss = test_total_loss / len(test_dataset)
    test_accuracy = test_correct / len(test_dataset)
    print(
        f"experiment: {args.experiment}, "
        f"test_loss: {test_avg_loss:.4f}, "
        f"test_accuracy: {test_accuracy:.2%}"
    )


if __name__ == "__main__":
    main()
