import torch
from torch import nn,load
from torchmetrics import Accuracy
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

from number_classifier import NumberClassifier

# Get testing data
test = datasets.MNIST(
    root="test_data", download=True, train=False, transform=ToTensor()
)
test_dataloader = DataLoader(test, 32)

# Create instance of our Classifier and choose our loss func
clf = NumberClassifier().to("cuda")
loss_fn = nn.CrossEntropyLoss()

# setup Accuracy function
accuracy = Accuracy(task="multiclass", num_classes=10).to("cuda")

if __name__ == "__main__":
    with open("results\model_state.pt", "rb") as f:
        clf.load_state_dict(load(f))

    test_loss, test_acc = 0, 0
    clf.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test, y_test = X_test.to("cuda"), y_test.to("cuda")
            # Forward pass
            test_pred = clf(X_test)

            # Calculate the loss
            loss = loss_fn(test_pred, y_test)
            test_loss += loss

            # Calc Accuracy
            test_acc += accuracy(y_test, test_pred.argmax(dim=1))

        # calculate the test loss average per batch
        test_loss /= len(test_dataloader)

        # Calculate the test acc averatge per batch
        test_acc /= len(test_dataloader)

        print(
            f"\nTest loss; {test_loss:.4f}, Test acc: {test_acc:.4f}"
        )
