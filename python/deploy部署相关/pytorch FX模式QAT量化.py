import torch
import copy
from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx
import torch.quantization.observer as observer

#将模型转换为QAT版本模型，其他训练过程与正常训练类似
def qat_version_model(model):
    qconfig_dict = {
        # Global Config
        # use 'fbgemm' for server inference and
        # 'qnnpack' for mobile inference
        "":torch.ao.quantization.get_default_qat_qconfig('fbgemm'), #全局量化配置

        # # Disable by layer-name
        # "module_name": [(m, None) for m in disable_layers],

        # Or disable by layer-type
        "object_type": [
            (PositionalEmbedding, None),   #想要跳过量化层，可以设置为None
            (torch.nn.Softmax, softmax_qconfig), #指定与全局量化配置不同的量化配置
            # ......
        ],
    }
    model_to_quantize = copy.deepcopy(model)
    model_fp32_prepared = prepare_qat_fx(
        model_to_quantize, qconfig_dict)
    return model_fp32_prepared


qat_model = qat_version_model(float_model)

#********模型训练**********
ce_loss = torch.nn.CrossEntropyLoss()
learning_rate = 0.01
epoch_nums = 100
optimizer = torch.optim.SGD(qat_model.parameters(), momentum=0.1, lr=learning_rate)
for i in range(epoch_nums):
    for idx, (image, label) in enumerate(train_data_loader):
        res = qat_model(image)
        loss = ce_loss(res, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if idx % 100:
            print("epochs: {}, loss: ".format(i), loss.detach().numpy())


#********模型验证**********
quantized_model = convert_fx(qat_model)
evaluate(quantized_model, eval_data_loader)

# 保存量化版本的模型
# 以torchScript的形式
trace_model = torch.jit.trace(quantized_model, (input1, input2, input3))
torch.jit.save(trace_model, 'trace_model.pt')

#以onnx的形式
torch.onnx.export(quantized_model, (input1, input2, input3), "quantized_model.onnx",
input_names=['input1', 'input2', 'input3'], output_names=['result'],
opset_version=16, export_params=True)