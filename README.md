1、首先使用onnx_trans.py将onnx模型的输出维度进行调整，如v8目标检测模型的输出维度为（1，84，8400），转化为（1，8400，84）。
2、CMakelists文件已经编辑好，首先学习如何进行cmake编译c++工程。
CMakelists里面大概内容是：调用cuda、nvcc编译器，链接opencv库，生成precess可执行文件。
3、该项目是直接加载tensorrt转换的engine文件，所以该项目中不再使用tensorrt。所要使用的engine(trt)文件是tensorrt基于你的硬件生成的，不同的硬件设备是不一样的。
4、主要程序在propress.c中，注意修改engine文件路径和usb摄像头ID。
5、图像的预处理和后处理部分使用了cuda进行了加速，目前在jetson tx2上fps可以达到35。

