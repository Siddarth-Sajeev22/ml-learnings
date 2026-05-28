import torch 

# Example creating tensor from normal array 
data = [[1,1], [2,2]]
tensor = torch.tensor(data)

matA = torch.rand((3, 4))
matB = torch.rand((4, 5))

matC = matA @ matB
sum = matC.sum()

for row in matC : 
    print(row.mean())

x = torch.tensor(2.0, requires_grad = True)
y = torch.tensor(1.0, requires_grad = True)

res = x**2 + 3*x*y + y**3
res.backward()
x_grad = x.grad
y_grad = y.grad
print("x_grad:", x_grad, "y_grad:", y_grad)


def numerical_gradient(f, x, eps=1e-5): 
    x = x.detach().clone()
    grad_tensor = torch.zeros_like(x)
    for i in range(x.shape[0]) : 
        for j in range(x.shape[1]) :
            x_plus = x.clone(); x_plus[i][j] += eps
            x_minus = x.clone(); x_minus[i][j] -= eps 
            grad_tensor[i][j] = (f(x_plus) - f(x_minus)) / (2*eps)
    
    return grad_tensor 

def check_gradients(f, x): 
    x = x.detach().clone().requires_grad_(True)
    if x.grad is not None :
        x.grad.zero_()
    num_grad = numerical_gradient(f, x.detach())
    res = f(x)
    res.backward()
    auto_grad = x.grad.detach().clone()
    diff = (auto_grad - num_grad).abs().max().item()
    print(f"Max gradient difference: {diff:.2e}")
    assert diff < 1e-4, f"Gradient check failed: {diff}"
    print("Gradient check passed.")


x = torch.randn(3, 4, dtype=torch.float64)
check_gradients(lambda t: (t ** 2).sum(), x)
check_gradients(lambda t: torch.relu(t).sum(), x)



