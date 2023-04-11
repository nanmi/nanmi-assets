import tensorrt as trt

class MyPlugin(trt.IPluginV2DynamicExt):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def get_plugin_type(self):
        return "MyPlugin"

    def get_plugin_version(self):
        return "1"

    def get_output_dtype(self, index, input_types, output_types):
        return trt.DataType.FLOAT

    def get_output_dimensions(self, index, input_dims, output_dims):
        output_dims = input_dims
        return output_dims

    def configure_plugin(self, input_shapes, plugin_config):
        self.input_shape = input_shapes[0]

    def serialize(self):
        return b""

    def create_plugin(self, name, inputs, outputs, config):
        return self

    def enqueue(self, batch_size, inputs, outputs, workspace, stream):
        # Implement plugin logic here
        pass

# Create plugin registry
plugin_registry = trt.get_plugin_registry()


# Define plugin creator function
def create_my_plugin(param1, param2):
    return MyPlugin(param1, param2)

# Register custom plugin
plugin_registry.register_creator("MyPlugin", create_my_plugin)




import tensorrt as trt

# Register custom plugin
plugin_creator = trt.PluginCreator()
plugin_creator.add_plugin_v2("MyPlugin", MyPlugin)

# Create plugin
plugin = plugin_creator.create_plugin("MyPlugin", {"param1": 1, "param2": 2})

# Create builder
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))

# Set maximum batch size
builder.max_batch_size = 1

# Set maximum workspace size
builder.max_workspace_size = 1 << 30 # G


# Use plugin in TensorRT engine
network = builder.create_network()
input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 3, 224, 224))
plugin_tensor = network.add_plugin_v2(inputs=[input_tensor], plugin=plugin)
output_tensor = network.add_output(name="output", tensor=plugin_tensor.get_output(0))
engine = builder.build_cuda_engine(network)