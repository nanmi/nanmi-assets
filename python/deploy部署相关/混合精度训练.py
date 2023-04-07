'''
混合精度训练是一种在训练过程中使用单精度和半精度浮点数进行计算的技术。这可以显著减少训练的内存使用和计算成本，同时仍然保持准确性。

在PyTorch中，可以使用torch.cuda.amp模块实现混合精度训练。该模块提供了一个名为autocast的上下文管理器，它可以将某些操作自动转换为半精度浮点数。
此外，该模块还提供了GradScaler类，可用于在反向传播过程中缩放梯度，以防止下溢或溢出。
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Define your model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

model = MyModel().cuda()

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define your data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define your scaler
scaler = GradScaler()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Perform the forward pass
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Perform the backward pass and update the weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Print the loss
        if i % 100 == 0:
            print(f"Epoch {epoch}, Batch {i}: Loss = {loss.item()}")
