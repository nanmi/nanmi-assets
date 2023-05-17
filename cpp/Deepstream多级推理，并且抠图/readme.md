# deepstream 多级推理示例

## 一二级模型介绍
### 一级模型
使用yolov5s 6.0 只检测人的模型，只有一个类别
输入 1x3x640x640
输出 6001x1x1

### 二级模型
使用MobileNetv3训练的分类模型，只有3各类别：green other red，必须按照这个顺序排列
输入 1x3x224x224
输出 1x3, 增加了softmax作为输出，官方没有接softmax

## Note
在后处理nvdsinfer_custom_impl_plugin/nvdsparser_custom_impl.cpp中的解析bbox等信息时将高度裁剪成上一半方便后面二级模型分类，这个可以生效已经测试过了，二级nvinfer插件中的gst_nvinfer_process_objects函数中在get_converted_buffer函数调用之前，输出第四个参数&object_meta->rect_params信息发现高度已经成一半了,而且如果不在osd渲染前高度重新调整则会渲染只有上半部分

在NvDsInferParseMobilenetv3函数中一定要使用strdup否则会报错Segmentation fault

## 运行
```
deepstream-app -c deepstream_app_config.txt
```
