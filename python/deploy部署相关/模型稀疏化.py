import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SparseModel(nn.Module):
    def __init__(self):
        super(SparseModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# 方法一
# Create a sparse model by setting some of the weights to zero
model = SparseModel()
for name, param in model.named_parameters():
    if 'weight' in name:
        # Set 50% of the weights to zero
        param.data = torch.from_numpy(np.random.binomial(1, 0.5, size=param.data.shape)).float() * param.data

# 方法二
# Create a sparse model by pruning the weights
model = SparseModel()
import torch.nn.utils.prune as prune
prune.l1_unstructured(model.fc1, name='weight', amount=0.5)
prune.l1_unstructured(model.fc2, name='weight', amount=0.5)

# 方法三
# Create a sparse model using L1 regularization
model = SparseModel()
num_epochs = ...
train_loader = ...
device = ...
alpha = ...
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Add L1 regularization to the loss function
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'weight' in name:
                l1_reg = l1_reg + torch.norm(param, 1)
        loss = loss + alpha * l1_reg

        loss.backward()
        optimizer.step()


# Save the sparse model to a file
torch.save(model.state_dict(), 'sparse_model.pt')

# Export the sparse model to an ONNX file
input_shape = (1, 784)
dummy_input = torch.randn(input_shape)
output_path = 'sparse_model.onnx'
torch.onnx.export(model, dummy_input, output_path, opset_version=11)


'''
在某些情况下，可能无法创建稀疏模型。例如，如果模型具有非常少量的参数，或者如果参数已经非常稀疏，那么在不显著降低模型性能的情况下，
可能不可能进一步稀疏。此外，某些类型的层，例如具有小滤波器大小的卷积层，可能很难在不引入伪影或降低模型精度的情况下稀疏化。
对于每个特定的用例，仔细评估稀疏性和模型性能之间的权衡是很重要的。

稀疏模型是通过直接处理训练模型的权重来创建的，而不需要输入数据。稀疏性是通过将一些权重设置为零来实现的，这可以使用各种技术（如修剪或正则化）来实现。
一旦确定了稀疏性模式，就可以使用前面描述的方法来保存和加载稀疏模型，而不需要输入数据。然而，需要注意的是，如果对模型进行微调或对新数据进行重新训练，
则可能需要重新确定稀疏性模式。


'''



