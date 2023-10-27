from torch import nn, save
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm

from number_classifier import NumberClassifier


# Get training data
train = datasets.MNIST(
    root="train_data", download=True, train=True, transform=ToTensor()
)
train_dataloader = DataLoader(train, 32)

# Instance of the neural network, loss, optimizer
clf = NumberClassifier().to("cuda")
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
epochs = 10

# Training flow
if __name__ == "__main__":
    for epoch in tqdm(range(epochs)):
        
        train_loss = 0
        for batch, (X,y) in enumerate(train_dataloader):
            X,y = X.to('cuda'), y.to('cuda')
            y_pred = clf(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            
            # Apply backpropagation
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        train_loss /= len(train_dataloader)     # batch loss
        print(f"\n Epoch:{epoch} loss is {train_loss}")
    
    # After the training - safe the trained model
    with open('results\model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)