from torch.utils.data import DataLoader 
from torchvision import datasets, transforms 
import torch.nn as nn 
import torch 

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

class MLP(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x): 
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (preds.argmax(dim=1) == y_batch).sum().item()
        total += y_batch.size(0)
    print(f"Epoch {epoch+1:2d} | loss: {total_loss/len(train_loader):.4f} | train acc: {correct/total:.4f}")

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        predicted = preds.argmax(dim=1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

print(f"Test accuracy: {correct / total:.4f}")
torch.save(model.state_dict(), 'mnist_mlp.pt')





