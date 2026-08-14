import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False, download=True,
    transform= torchvision.transforms.ToTensor()
)
test_datasize = len(test_dataset)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=False)

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


model = myNet()
state_dict = torch.load(
    "./checkpoints/best_model.pth",
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)

model.to(device)

loss_fn = CrossEntropyLoss()
loss_fn.to(device)

model.eval()
with torch.no_grad():
    test_total_loss = 0
    test_accuracy = 0
    for imgs,labels in test_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = model(imgs)
        loss = loss_fn(output, labels)
        test_total_loss += loss.item() * imgs.size(0)
        test_accuracy += (output.argmax(1) == labels).sum().item()
test_avg_loss = test_total_loss / test_datasize
test_accuracy = test_accuracy / test_datasize
print("test_avg_loss:{},test_accuracy:{}".format(test_avg_loss,test_accuracy))