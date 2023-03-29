import torch.nn as nn
import torch.nn.utils.prune as prune

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        x = self.linear(x)
        return x

model = MyModel()

# 非结构化裁剪
# Prune 50% of the weights in the linear layer
prune.l1_unstructured(model.linear, name='weight', amount=0.5)
# Remove the pruned weights from the model
prune.remove(model.linear, 'weight')

# 全局非结构化裁剪
# Prune 50% of the weights in the entire model using global unstructured pruning
prune.global_unstructured(
    parameters=[(model.linear1, 'weight'), ...],
    pruning_method=prune.L1Unstructured,
    amount=0.5,
)
# Remove the pruned weights from the model
prune.remove(model.linear1, 'weight')

# 结构化裁剪
# Prune 50% of the channels in the convolutional layer using L1 unstructured pruning
prune.l1_unstructured(model.conv, name='weight', amount=0.5, dim=1)
# Remove the pruned channels from the model
prune.remove(model.conv, 'weight')






'''
要在PyTorch中执行模型修剪，可以使用PyTorch提供的修剪模块。该模块提供了修剪模型的各种方法，例如修剪单个权重、整个通道，甚至整个层。
通常，模型修剪包括从模型中删除某些参数，以减小模型的大小并提高其效率。修剪的模型的特定部分取决于所使用的修剪方法。
例如，在前面提供的代码片段中，我们将L1非结构化修剪应用于线性层的权重参数，指定要修剪50%的权重。这意味着线性层的权重参数中的一半权重将从模型中删除。
修剪模块提供的其他修剪方法，例如L2非结构化修剪、全局非结构化修剪或结构化修剪，可以修剪模型的不同部分。

非结构化修剪包括从模型中移除单个权重，而不考虑它们在模型中的位置。这可以使用修剪模块提供的l1_structured或l2_ustructureed方法来完成。在前面提供的示例中，我们将L1非结构化修剪应用于线性层的权重参数，指定要修剪50%的权重。
全局非结构化修剪包括从整个模型中移除固定百分比的权重，而不考虑它们在模型中的位置。这可以使用prune模块提供的global_ustructured方法来完成。
结构化修剪包括从模型中移除整个通道或层，而不是单个权重。这可以使用修剪模块提供的l1_ustructured或l2_ustructued方法来完成，指定dim参数来指示要修剪的维度。在前面提供的示例中，我们将L1非结构化修剪应用于conv层的权重参数，指定要修剪50%的通道。

需要明确的是，从技术上讲，任何模型都可以进行修剪，但在某些情况下，修剪可能无效，甚至可能损害模型的性能。例如，如果一个模型已经很小或参数数量很低，那么修剪可能不会带来太多好处。此外，如果模型已经过拟合训练数据，修剪可能会降低模型的容量，从而加剧这个问题。
还值得注意的是，某些类型的层，如批处理规范化等规范化层，可能不会被直接修剪，因为它们不包含任何可训练的参数。相反，可以通过修剪它们之前或之后的层来间接修剪这些层。
总的来说，在决定是否进行修剪之前，仔细考虑模型的具体特征和修剪的目标是很重要的。

除了线性和卷积层之外，可以修剪的其他类型的层包括：

-RNN层（例如“nn.LSTM”、“nn.GRU”）
-Embedding层（例如“nn.Embedding”）
-Transformer层（例如“nn.TransferEncoder”、“nn.TransformerDecoder”）
-从`nn.Module继承的任何自定义层`

如果你指的是L1和L2正则化之间的差异，L1正则化通过将权重的绝对值添加到损失函数来鼓励权重的稀疏性，而L2正则化通过向损失函数添加权重的平方来鼓励小权重。
在修剪的上下文中，L1非结构化修剪去除具有最小绝对值的一定百分比的权重，而L2非结构化修剪移除具有最小平方值的一定比例的权重。
总体而言，L1和L2修剪之间的选择可能取决于模型的特定特性和修剪的目标。L1修剪对于实现稀疏性可能更有效，而L2修剪对于保留权重的总体大小可能更有效。然而，值得注意的是，这两种方法都可以有效地缩小模型的大小并提高其效率。
'''

# 对于Batch Normal裁剪
'''
要在PyTorch中执行批量规范化修剪，可以使用PyTorch提供的修剪模块。然而，批处理规范化通常不会直接修剪，因为它是一个不包含任何可训练参数的规范化层。相反，批处理规范化通常是通过修剪之前或之后的层来间接修剪的。
例如，如果要修剪批处理规范化层之前的卷积层，可以使用修剪模块提供的l1_structured或l2_ustructureed方法，指定name参数来指示要修剪的参数。例如，要使用L1非结构化修剪来修剪卷积层中50%的权重，可以执行以下操作：
'''
import torch.nn as nn
import torch.nn.utils.prune as prune

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 6, 5)
        self.bn = nn.BatchNorm2d(6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

model = MyModel()

# Prune 50% of the weights in the convolutional layer using L1 unstructured pruning
prune.l1_unstructured(model.conv, name='weight', amount=0.5)

# Remove the pruned weights from the model
prune.remove(model.conv, 'weight')

