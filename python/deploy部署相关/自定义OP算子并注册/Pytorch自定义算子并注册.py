import torch
from torch.nn import functional as F
from torch.autograd import Function

class CustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        assert input1.size() == input2.size()
        output = torch.sum(input1 * input2)
        ctx.save_for_backward(input1, input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = grad_input2 = None
        if ctx.needs_input_grad[0]:
            grad_input1 = grad_output * input2
        if ctx.needs_input_grad[1]:
            grad_input2 = grad_output * input1
        return grad_input1, grad_input2

def custom_op(input1, input2):
    return CustomOp.apply(input1, input2)

# Register the operator
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn import init

class CustomModule(Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.weight = Parameter(torch.Tensor(1))

    def forward(self, input1, input2):
        return custom_op(input1, input2) * self.weight



def unit_test():
    x = torch.ones(3, 3)
    y = torch.ones(3, 3) * 2
    z = CustomModule()
    z.weight.data.fill_(3)
    output = z(x, y)
    print(output) # should output 54

if __name__ == "__main__":
    unit_test()