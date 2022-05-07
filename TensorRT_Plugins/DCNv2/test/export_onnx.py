import torch
from dcn_v2 import DCN, DCNPooling, DCNv2, DCNv2Pooling

def export_dconv_onnx():
    input = torch.randn(1, 64, 128, 128).cuda()
    # wrap all things (offset and mask) in DCN
    dcn = DCN(64, 64, kernel_size=(3, 3), stride=1, padding=1, deformable_groups=2).cuda()
    dcn.eval()
    dynamic = False
    torch.onnx.export(
    dcn,
    input,
    "dcn.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: 'batch'},
                    "output": {0: 'batch'}} if dynamic else None,
    opset_version=11,
    do_constant_folding=True
    )
    print("export dcn onnx successfully!")

if __name__ == "__main__":
    export_dconv_onnx()
    