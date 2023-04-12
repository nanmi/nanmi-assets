1.生成engine文件

2.将engine文件重命名成`model.plan` 放在`/models/yolov5s_t/1/`

3.在`/models/yolov5s_t/`下新建模型配置管理文件`config.pbtxt`

4.在`/plugins/yolov5s_t/`下放置插件`libyolo.so`

5.一切准备好后，命令行运行`LD_PRELOAD=/plugins/yolov5s_t/libyolo.so tritonserver --model-repository=/models`
