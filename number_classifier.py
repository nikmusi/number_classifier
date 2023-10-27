from torch import nn

# The Number Classifier (simple CNN)
class NumberClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)     # CNN for MINST with img size 28x28
        )
    
    def forward(self, x):
        return self.model(x)

        
    
    
    