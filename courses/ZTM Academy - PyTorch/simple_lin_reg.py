from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim


print(torch.__version__)
torch.manual_seed(29)
torch.cuda.manual_seed(29)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('GPU' if device == 'cuda' else 'CPU')

# Data preparation (put the data into device)
weight = 0.7
bias = 0.3
X = torch.arange(0, 1, step=0.02).unsqueeze(dim=1).to(device=device)
y = bias + weight * X

# Split the data
n = len(X)
split = int(n * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# Build a model
class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


model = LinearRegressionModel()
model.to(device=device)

# Train the model
loss_fn = nn.L1Loss()  # MAE loss function
optimizer = optim.SGD(params=model.parameters(), lr=1e-2)

train_loss_values: list[float] = []
test_loss_values: list[float] = []
epochs = 1000

for i in tqdm(range(1, epochs + 1), desc='Model training process'):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Model Evaluation
    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, y_test)

    train_loss_values.append(loss.item())
    test_loss_values.append(test_loss.item())

for name, param in model.named_parameters():
    print(name, param.data)

models_folder = Path('../../models/')
models_folder.mkdir(exist_ok=True, parents=True)
model_path = models_folder / 'linear_regression_model.pth'
torch.save(obj=model.state_dict(), f=model_path)

plt.plot(train_loss_values, label='Train loss')
plt.plot(test_loss_values, label='Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss values')
plt.title('Loss decreasing of train and test data')
plt.legend()
plt.grid()
plt.show()
