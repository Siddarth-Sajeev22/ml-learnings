from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torch

X_np, y_np = make_moons(n_samples=200, noise=0.2, random_state=42)
X_np, X_test_np , y_np, y_test_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42)
X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)  # shape (200, 1)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)


# intialize weights 
W1 = torch.randn(2, 8) * 0.1
b1 = torch.zeros(1, 8)
W2 = torch.randn(8, 1) * 0.1
b2 = torch.zeros(1, 1)
N = X.shape[0]

def relu(x): 
    return torch.clamp(x, min=0)

def sigmoid(x): 
    return 1/(1 + torch.exp(-x))

def forward_pass(X, W1, b1, W2, b2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2 
    a2 = sigmoid(z2)
    return a2 , a1, z2, z1

def loss(y, a2): 
    loss = -1 * (y*torch.log(a2) + (1-y)*torch.log(1-a2)).mean()
    return loss 

def backward_pass(y, a2, a1, z1, N):
    dL_da2 = -(y/a2 - (1-y)/(1-a2)) / N
    dL_dz2 = dL_da2 * a2 * (1-a2)
    dL_dw2 = a1.T @ dL_dz2
    dL_db2 = dL_dz2.sum(axis = 0)
    dL_da1 = dL_dz2 @ W2.T
    dL_dz1 = dL_da1 * (z1 > 0).float()
    dL_dw1 = X.T @ dL_dz1
    dL_db1 = dL_dz1.sum(axis = 0)
    return dL_dw2 , dL_db2, dL_dw1, dL_db1

def main(): 
    global W1, b1, W2, b2, N
    lr = 0.5
    for i in range(3000):
        a2, a1, z2, z1 = forward_pass(X, W1, b1, W2, b2)
        l = loss(y, a2)
        dL_dw2 , dL_db2, dL_dw1, dL_db1 = backward_pass(y, a2, a1, z1, N)
        W1 -= lr * dL_dw1 
        b1 -= lr * dL_db1
        W2 -= lr * dL_dw2
        b2 -= lr * dL_db2

        if i % 100 == 0:
            print(f"step {i}, loss: {l:.4f}")
    

    
    preds = (a2 > 0.5) * 1.0
    accuracy = ((preds == y) * 1.0).mean()
    print(f"Train Accuracy: {accuracy:.4f}")

    a2_test, _, _, _ = forward_pass(X_test, W1, b1, W2, b2)
    test_preds = (a2_test > 0.5) * 1.0
    test_accuracy = ((test_preds == y_test) * 1.0).mean()
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__": 
    main()
