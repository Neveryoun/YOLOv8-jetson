#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferPlugin.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <string>

#include "test.h"  //cuda

using namespace nvinfer1;
using namespace cv;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int CLASSES = 80;
static const int Num_box = 8400;
static const int OUTPUT_SIZE = Num_box * (CLASSES+4);//output0

static const float CONF_THRESHOLD = 0.1;
static const float NMS_THRESHOLD = 0.5;
static const float MASK_THRESHOLD = 0.5;
const char* INPUT_BLOB_NAME = "images";       //这里对应模型的输入和输出名称，通过onnx文件在netron中可查看
const char* OUTPUT_BLOB_NAME = "output0";


struct OutputSeg {
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
	cv::Mat boxMask;       //矩形框内mask，节省内存空间和加快速度
};

void DrawPred(Mat& img,std:: vector<OutputSeg> result) {
	std::vector<Scalar> color;
	srand(time(0));
	color.push_back(Scalar(0, 255, 0));

	//Mat mask = img.clone();
	for (int i = 0; i < result.size(); i++) {
		int left, top;
		left = result[i].box.x;
		top = result[i].box.y;
		int color_num = i;
		rectangle(img, result[i].box, color[0], 2, 8);
		
		char label[100];
		sprintf(label, "%d:%.2f", result[i].id, result[i].confidence);

		//std::string label = std::to_string(result[i].id) + ":" + std::to_string(result[i].confidence);
		int baseLine;
		Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		top = max(top, labelSize.height);
		putText(img, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, color[0], 2);
	}
	
	//addWeighted(img, 0.5, mask, 0.8, 1, img); //将mask加在原图上面
}

static Logger gLogger;
void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));//
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
	//CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float)));
	// cudaMalloc分配内存 cudaFree释放内存 cudaMemcpy或 cudaMemcpyAsync 在主机和设备之间传输数据
	// cudaMemcpy cudaMemcpyAsync 显式地阻塞传输 显式地非阻塞传输 
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
	//CHECK(cudaFree(buffers[outputIndex1]));
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		argv[1] = "/home/zeh/Desktop/cpp_tensorrt/models/yolo/yolov8n1_fp16.engine";
	}
	// create a model using the API directly and serialize it to a stream
	char* trtModelStream{ nullptr }; //char* trtModelStream==nullptr;  开辟空指针后 要和new配合使用
	size_t size{ 0 };//与int固定四个字节不同有所不同,size_t的取值range是目标平台下最大可能的数组尺寸,一些平台下size_t的范围小于int的正数范围,又或者大于unsigned int. 使用Int既有可能浪费，又有可能范围不够大。

	std::ifstream file(argv[1], std::ios::binary);
	if (file.good()) {
		std::cout << "load engine success" << std::endl;
		file.seekg(0, file.end);//指向文件的最后地址
		size = file.tellg();//把文件长度告诉给size
		//std::cout << "\nfile:" << argv[1] << " size is";
		//std::cout << size << "";

		file.seekg(0, file.beg);//指回文件的开始地址
		trtModelStream = new char[size];//开辟一个char 长度是文件的长度
		assert(trtModelStream);//
		file.read(trtModelStream, size);//将文件内容传给trtModelStream
		file.close();//关闭
	}
	else {
		std::cout << "load engine failed" << std::endl;
		return 1;
	}

	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
	assert(engine != nullptr);
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);
	delete[] trtModelStream;
	
	VideoCapture cap(3);
	if(!cap.isOpened()){
		std::cout<<"creamer error!"<<std::endl;
		return -1;
	}
	
	double fps;
	while(true){
		Mat src;
		cap.read(src);

	// Mat src = imread(argv[2], 1);
	// if (src.empty()) { std::cout << "image load faild" << std::endl; return 1; }
	int img_width = src.cols;
	int img_height = src.rows;
	//std::cout << "宽高：" << img_width << " " << img_height << std::endl;
	// Subtract mean from image
	//static float data[3 * INPUT_H * INPUT_W];
	Mat pr_img0, pr_img;
	std::vector<int> padsize;
	
	auto start_fps = std::chrono::system_clock::now();
	auto start = std::chrono::system_clock::now();
	pr_img = preprocess_img(src, INPUT_H, INPUT_W, padsize);       // Resize
	int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
	float ratio_h = (float)src.rows / newh;
	float ratio_w = (float)src.cols / neww;
	int i = 0;// [1,3,INPUT_H,INPUT_W]
	//std::cout << "pr_img.step" << pr_img.step << std::endl;

	static float data[3 * INPUT_H * INPUT_W];
	size_t dataBytes = INPUT_H*INPUT_W*3;

	uchar* imageData;
	cudaMalloc((void**)&imageData,sizeof(uchar)*dataBytes);
	float* deviceData;
	cudaMalloc((void**)&deviceData,dataBytes*sizeof(float));

	cudaMemcpy(imageData, pr_img.data, dataBytes*sizeof(uchar), cudaMemcpyHostToDevice);
	//cudaMemcpy(deviceData, data, dataBytes, cudaMemcpyHostToDevice);

	mycuda(imageData, deviceData, INPUT_H, INPUT_W);
    cudaMemcpy(data, deviceData, dataBytes*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceData);

	// for (int row = 0; row < INPUT_H; ++row) {
	// 	uchar* uc_pixel = pr_img.data + row * pr_img.step;//pr_img.step=widthx3 就是每一行有width个3通道的值
	// 	for (int col = 0; col < INPUT_W; ++col)
	// 	{

	// 		data[i] = (float)uc_pixel[2] / 255.0;
	// 		data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
	// 		data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
	// 		uc_pixel += 3;
	// 		++i;
	// 	}
	// }
	auto end = std::chrono::system_clock::now();
	std::cout << "预处理时间" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< "ms" << std::endl;

	
	// Run inference
	static float prob[OUTPUT_SIZE];
	
	start = std::chrono::system_clock::now();
	//auto start_fps = std::chrono::system_clock::now();
	doInference(*context, data, prob, 1);
	end = std::chrono::system_clock::now();
	std::cout << "推理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	

	

	// 处理box
	int net_length = CLASSES + 4;
	cv::Mat out1 = cv::Mat(net_length, Num_box, CV_32F, prob);

	start = std::chrono::system_clock::now();
	for (int i = 0; i < Num_box; i++) {
		//输出是1*net_length*Num_box;所以每个box的属性是每隔Num_box取一个值，共net_length个值
		cv::Mat scores = out1(Rect(i, 4, 1, CLASSES)).clone();
		Point classIdPoint;
		double max_class_socre;
		minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
		max_class_socre = (float)max_class_socre;
		if (max_class_socre >= CONF_THRESHOLD) {
			
			float x = (out1.at<float>(0, i) - padw) * ratio_w;  //cx
			float y = (out1.at<float>(1, i) - padh) * ratio_h;  //cy
			float w = out1.at<float>(2, i) * ratio_w;  //w
			float h = out1.at<float>(3, i) * ratio_h;  //h
			int left = MAX((x - 0.5 * w), 0);
			int top = MAX((y - 0.5 * h), 0);
			int width = (int)w;
			int height = (int)h;
			if (width <= 0 || height <= 0) { continue; }

			classIds.push_back(classIdPoint.y);
			confidences.push_back(max_class_socre);
			boxes.push_back(Rect(left, top, width, height));
		}

	}
	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);
	std::vector<cv::Mat> temp_mask_proposals;
	std::vector<OutputSeg> output;
	Rect holeImgRect(0, 0, src.cols, src.rows);
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		OutputSeg result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx]& holeImgRect;
		output.push_back(result);
		//temp_mask_proposals.push_back(picked_proposals[idx]);
	}

	end = std::chrono::system_clock::now();
	std::cout << "后处理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<< "ms" << std::endl;

	DrawPred(src, output);
	cv::imshow("output1.jpg", src);
	auto end_fps = std::chrono::system_clock::now();
    fps = std::chrono::duration_cast<std::chrono::milliseconds>(end_fps - start_fps).count() ;
	fps=1000/fps;
	std::cout<<fps<<std::endl;
	//start0=end0;
	if(cv::waitKey(1)=='q'){
		break;
	}
	//char c = cv::waitKey(0);
	//Destroy the engine
    //context->destroy();
    //engine->destroy();
    //runtime->destroy();
	}
	system("pause");
    return 0;
}
