1、CMakelists文件已经编辑好，首先学习如何进行cmake编译c++工程。
CMakelists里面大概内容是：调用cuda、nvcc编译器，链接opencv库，生成propress可执行文件。
2、该项目是直接加载tensorrt转换的engine文件，所以该项目中不再使用tensorrt。所要使用的engine(trt)文件是tensorrt基于你的硬件生成的，不同的硬件设备是不一样的。
3、主要程序在propress.c中，注意修改engine文件路径和usb摄像头ID。
3、图像的预处理和后处理部分使用了cuda进行了加速，目前fps可以达到35。

