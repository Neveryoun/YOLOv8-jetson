1、CMakelists文件已经编辑好，首先学习如何进行cmake编译c++工程。
CMakelists里面大概内容是：调用cuda、nvcc编译器，链接opencv库，生成trt_infer可执行文件。
2、该项目是直接加载tensorrt转换的engine文件，所以该项目中不再使用tensorrt。例程中的engine文件在model_engine文件夹中。
3、主要程序在main2_trt_infer中，注意修改engine文件路径和usb摄像头ID。
