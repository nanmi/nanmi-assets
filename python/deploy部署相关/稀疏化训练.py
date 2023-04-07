'''
一个重要的考虑因素是修剪方法的选择。在前面提供的示例中，使用了l1_structured方法，
该方法删除了网络中的最小权重。然而，PyTorch中还有其他可用的修剪方法，如ln_structed
和random_ustructured，它们可能更适合某些模型或任务。您可以在PyTorch文档中找到有关
这些方法的更多信息。

另一个重要的考虑因素是稀疏性对训练过程的影响。稀疏模型可能需要更长的训练时间或更小的
学习率才能实现与密集模型相当的性能。此外，稀疏性级别的选择可能取决于特定的任务和数据
集，并且可能需要一些实验来找到最佳值。

最后，值得注意的是，稀疏训练也可以应用于模型的其他方面，而不仅仅是权重，例如激活或
梯度。这可以进一步降低训练和推理的计算成本。
'''

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

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

model = MyModel()

# Define the pruning parameters
pruning_params = {
    'name': 'weight',
    'pruning_method': 'l1_unstructured',
    'amount': 0.5,
}

# Apply pruning to the model
prune.global_unstructured(
    [model.fc1, model.fc2],
    pruning_method=pruning_params['pruning_method'],
    amount=pruning_params['amount'],
    n = 2, # n＝2，这意味着修剪是沿着权重张量的第二维度（即，权重矩阵的列）,默认情况下，n＝0，这意味着修剪是沿着权重张量的第一维度（即，权重矩阵的行）
)

# Train the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

train_loader = ...
num_epochs = 20
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


