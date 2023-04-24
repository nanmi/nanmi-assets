
'''
在模型提取中，我们计算教师模型的输出和学生模型的输出的softmax，
因为我们想将教师模型提供的“软”目标与学生模型做出的“软的”预测进行比较。
softmax函数用于将模型的输出转换为类上的概率分布。在应用softmax函数之前，
通过将每个模型的输出除以温度参数T，我们可以控制目标和预测的“柔和度”。
较高的温度参数将导致较软的目标和预测，而较低的温度参数则会导致较硬的目标和预报。
通过使用软目标，我们可以鼓励学生模型从教师模型的行为中学习，即使教师模型的输出
不是一个热点向量。这可以帮助学生模型更好地泛化，避免对训练数据的过度拟合。

通常，温度参数T的选择取决于具体任务以及教师和学生模型的特性。当教师模型对其预测
非常有信心时，更高的温度参数可能是有用的，因为即使教师模型的预测不完全准确，
它也可以帮助学生模型从教师模型的行为中学习。另一方面，当教师模型对其预测不太有
信心时，较低的温度参数可能是有用的，因为它可以帮助学生模型专注于教师模型行为的
最重要方面。

在所提供的代码中，温度参数T被设置为20。这个值可能是基于经验实验选择的，可能不是
所有任务和模型的最佳值。建议尝试不同的T值，并在验证集上评估学生模型的性能，以
确定手头特定任务的最佳值。
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define teacher and student models
teacher_model = ...
student_model = ...
num_epochs = ...
dataloader = ...

# Define loss function,也可以用KL散度来代替
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(student_model.parameters(), lr=1e-3)

# Define temperature parameter
T = 20

# Train student model using distillation
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # Compute softmax of teacher model's output divided by T
        teacher_outputs = F.softmax(teacher_model(inputs) / T, dim=1)
        
        # Compute softmax of student model's output divided by T
        student_outputs = F.softmax(student_model(inputs) / T, dim=1)
        
        # Compute MSE loss between softmax outputs of teacher and student models
        loss = criterion(student_outputs, teacher_outputs)
        
        # Backpropagate loss and update student model's parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save student model
torch.save(student_model.state_dict(), 'student_model.pth')

