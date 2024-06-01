#include <opencv2/opencv.hpp>
#include "test.h"
#include <chrono>
using namespace cv;

const char *cocolabels[] = {"person",        "bicycle",      "car",
                                   "motorcycle",    "airplane",     "bus",
                                   "train",         "truck",        "boat",
                                   "traffic light", "fire hydrant", "stop sign",
                                   "parking meter", "bench",        "bird",
                                   "cat",           "dog",          "horse",
                                   "sheep",         "cow",          "elephant",
                                   "bear",          "zebra",        "giraffe",
                                   "backpack",      "umbrella",     "handbag",
                                   "tie",           "suitcase",     "frisbee",
                                   "skis",          "snowboard",    "sports ball",
                                   "kite",          "baseball bat", "baseball glove",
                                   "skateboard",    "surfboard",    "tennis racket",
                                   "bottle",        "wine glass",   "cup",
                                   "fork",          "knife",        "spoon",
                                   "bowl",          "banana",       "apple",
                                   "sandwich",      "orange",       "broccoli",
                                   "carrot",        "hot dog",      "pizza",
                                   "donut",         "cake",         "chair",
                                   "couch",         "potted plant", "bed",
                                   "dining table",  "toilet",       "tv",
                                   "laptop",        "mouse",        "remote",
                                   "keyboard",      "cell phone",   "microwave",
                                   "oven",          "toaster",      "sink",
                                   "refrigerator",  "book",         "clock",
                                   "vase",          "scissors",     "teddy bear",
                                   "hair drier",    "toothbrush"};


int main(int argc, char** argv){

    argv[1] = "/home/zeh/Desktop/cpp_tensorrt/models/yolo/yolov8ntr.engine";
    //VideoCapture cap("/home/zeh/Desktop/cpp_tensorrt/output15.mp4");
    VideoCapture cap(0);

    const char* INPUT_BLOB_NAME = "images";       //这里对应模型的输入和输出名称，通过onnx文件在netron中可查看
    const char* OUTPUT_BLOB_NAME = "output0";
    static const int INPUT_H = 640;
    static const int INPUT_W = 640;
    static const int CLASSES = 80;
    static const int Num_box = 8400;
    static const float CONF_THRESHOLD = 0.1;
    static const float NMS_THRESHOLD = 0.5;
    static const float MASK_THRESHOLD = 0.5;
    static const int OUTPUT_SIZE = Num_box * (CLASSES+4);//output0
    const int num_image=1;
    const int MAX_IMAGE_BOXES=1024;
    using BoxArray=std::vector<Box>;

    uint8_t* psrc_device = nullptr;
    float* pdst_norm_device=nullptr;
    float* affine_matrix_device=nullptr;
    float* output_array=nullptr;    //GPU用于存储解码筛选后的框信息
    void* buffers[2];

    char* trtModelStream=nullptr; //char* trtModelStream==nullptr;  开辟空指针后 要和new配合使用
	size_t size=0;//与int固定四个字节不同有所不同,size_t的取值range是目标平台下最大可能的数组尺寸,一些平台下size_t的范围小于int的正数范围,又或者大于unsigned int. 使用Int既有可能浪费，又有可能范围不够大。
	std::ifstream file(argv[1], std::ios::binary);
	if (file.good()) {
		std::cout << "load engine success" << std::endl;
		file.seekg(0, file.end);//指向文件的最后地址
		size = file.tellg();//把文件长度告诉给size

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

    static Logger gLogger;
    IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
	assert(engine != nullptr);
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);
    const ICudaEngine& engine1 = context->getEngine();
    const int inputIndex = engine1.getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine1.getBindingIndex(OUTPUT_BLOB_NAME);
    CHECK(cudaMalloc(&buffers[inputIndex], 1 * 3 * INPUT_H * INPUT_W * sizeof(float)));//
	CHECK(cudaMalloc(&buffers[outputIndex], 1 * OUTPUT_SIZE * sizeof(float)));

    cudaMalloc((void**)&affine_matrix_device, sizeof(float) * 6);
    cudaMalloc((void**)&pdst_norm_device,640*640*3*sizeof(float));
    CHECK(cudaMalloc(&output_array, 32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT* sizeof(float)));
    float prob[1 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT];

    Mat src;
    int i=0;
    BoxArray arrout;

    while(cap.read(src)){
    //Mat src;
	int img_width = src.cols;
	int img_height = src.rows;
    size_t src_size = img_width * img_height * 3;  
    cudaMalloc((void**)&psrc_device, img_width*img_height*3); 

    auto start = std::chrono::system_clock::now();
    Norm normalize_;
    normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
    vector<AffineMatrix> affine_matrixs(1);
    affine_matrixs[0].compute(make_tuple(img_width, img_height),
                   make_tuple(INPUT_H, INPUT_W));

    cudaStream_t stream0;
    CHECK(cudaStreamCreate(&stream0));
    CHECK(cudaMemcpyAsync(psrc_device, src.data, src_size, cudaMemcpyHostToDevice,stream0));
    CHECK(cudaMemcpyAsync(affine_matrix_device, affine_matrixs[0].d2i, sizeof(float)*6, cudaMemcpyHostToDevice,stream0));

	warp_affine_bilinear_and_normalize_plane(psrc_device,img_width * 3, img_width, img_height,(float*)buffers[inputIndex],640, 640,affine_matrix_device,114,normalize_,stream0);

    context->enqueue(1, buffers, stream0, nullptr);

    decode_kernel_invoker((float*)buffers[outputIndex],Num_box,CLASSES,84,CONF_THRESHOLD,NMS_THRESHOLD,affine_matrix_device,output_array,MAX_IMAGE_BOXES,stream0);
    CHECK(cudaMemcpyAsync(prob, output_array, 32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT* sizeof(float), cudaMemcpyDeviceToHost,stream0));
    cudaMemsetAsync(output_array, 0, 32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT * sizeof(float));
    
    cudaStreamSynchronize(stream0);
    cudaStreamDestroy(stream0);

    int count = min(MAX_IMAGE_BOXES, (int)*prob);
    for (int i = 0; i < count; ++i) {
    float *pbox = prob + 1 + i * NUM_BOX_ELEMENT;
    int label = pbox[5];
    int keepflag = pbox[6];
    if (keepflag == 1) {
        Box result_object_box(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
        arrout.emplace_back(result_object_box);
    }
    }
    for(auto &obj:arrout){
    uint8_t b, g, r;
    tie(b, g, r) = std::make_tuple(0, 255, 0);
    cv::rectangle(src, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                  cv::Scalar(b, g, r), 5);
    auto name = cocolabels[obj.class_label];
    auto caption = cv::format("%s %.2f", name, obj.confidence);
    int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
    cv::rectangle(src, cv::Point(obj.left - 3, obj.top - 33),
                 cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
    cv::putText(src, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    arrout.clear();
    }

    cv::imshow("Result1.jpg", src);
    auto end = std::chrono::system_clock::now();
    double fps = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    fps=1000/fps;
	std::cout<<fps<<std::endl;
    CHECK(cudaFree(psrc_device));
    if(cv::waitKey(1)=='q'){
        break;
    }

    }
    CHECK(cudaFree(affine_matrix_device));
    CHECK(cudaFree(output_array));
    CHECK(cudaFree(pdst_norm_device));
    delete[] trtModelStream;
    return 0;
}
