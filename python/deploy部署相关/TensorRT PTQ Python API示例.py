import tensorrt as trt
import numpy as np

# Load the ONNX model
with open("path/to/model.onnx", "rb") as f:
    onnx_model = f.read()

# Create a TensorRT engine from the ONNX model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
onnx_parser = trt.OnnxParser(TRT_LOGGER)
onnx_parser.parse(onnx_model)
network = onnx_parser.convert_to_trt_network()
builder = trt.Builder(TRT_LOGGER)
builder.max_batch_size = 1
builder.max_workspace_size = 1 << 30 # 1G
engine = builder.build_cuda_engine(network)

# Generate calibration data
calibration_data = np.random.rand(100, 3, 224, 224).astype(np.float32)
calibrator = trt.IInt8LegacyCalibrator()
calibrator.set_algorithm(trt.CalibrationAlgoType.ENTROPY_CALIBRATION)
calibrator.set_data(calibration_data)
engine.set_int8_mode(True)
engine.set_int8_calibrator(calibrator)

# Save the quantized engine
with open("path/to/quantized_engine.engine", "wb") as f:
    f.write(engine.serialize())


