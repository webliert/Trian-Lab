import openvino as ov
import torch
from openvino.runtime import Core, serialize

# load PyTorch model into memory
model = torch.jit.load("Exported_policy/walk_lite_40700.pt")

# convert the model into OpenVINO model
# 正确的输入形状: (batch_size, features) = (1, 750)
example = torch.randn(1, 750)
ov_model = ov.convert_model(model, example_input=(example,))

# compile the model for CPU device
core = ov.Core()
compiled_model = core.compile_model(ov_model, 'CPU')

# serialize the model
serialize(ov_model, "Exported_policy/openvino/walk_lite_40700.xml", "Exported_policy/openvino/walk_lite_40700.bin")
print("Model converted successfully!")
