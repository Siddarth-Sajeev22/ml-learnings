from torchvision import datasets, transforms 
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch 

transform = transforms.ToTensor()
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class MLP(nn.Module): 
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x): 
        x= x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        return self.fc3(x)
    
class CNN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x): 
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# make it switchable between cnn or mlp , and do a comparison would be better 
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10): 
    total_loss = 0
    correct = 0
    total = 0
    for X_batch , y_batch in train_loader: 
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (preds.argmax(dim = 1) == y_batch).sum().item()
        total += y_batch.size(0)
    print(f"Epoch {epoch+1:2d} | loss: {total_loss/len(train_loader):.4f} | train acc: {correct/total:.4f}")

model.eval()
correct = 0 
total = 0 
with torch.no_grad(): 
    for X_batch, y_batch in test_loader : 
        preds = model(X_batch)
        correct += (preds.argmax(dim = 1)  == y_batch).sum().item()
        total += y_batch.size(0)
print(f"Test accuracy: {correct/total:.4f}")


