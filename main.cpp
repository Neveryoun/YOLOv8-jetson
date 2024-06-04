#include <opencv2/opencv.hpp>
#include "yolo.h"
#include <chrono>
using namespace cv;
using BoxArray=std::vector<Box>;

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

    //VideoCapture cap("/home/zeh/Desktop/cpp_tensorrt/output15.mp4");
    VideoCapture cap(0);

    std::shared_ptr<Myyolo> v8=std::make_shared<Myyolo>("/home/zeh/Desktop/cpp_tensorrt/models/yolo/yolov8ntr.engine");
    Mat src;
    BoxArray arrout;

    while(cap.read(src)){
    auto start = std::chrono::system_clock::now();
    /*图像预处理部分(letterbox、归一化），预处理结果保存在buffers[0]中*/
    v8->preprocess(src);   
    /*图像推理部分，输入为buffers，最后将推理结果保存在buffers[1]中*/            
    v8->infer();    
    /*将buffers[1]的结果映射到原图上(做letterbox之前)，nms处理，根据置信度等处理，保存到prob中*/                   
    v8->decode_nms();     

    /*根据得到的prob在原图上画框*/
    int count = min(int(v8->MAX_IMAGE_BOXES), (int)*v8->prob);
    for (int i = 0; i < count; ++i) {
    float *pbox = v8->prob + 1 + i * NUM_BOX_ELEMENT;
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
    if(cv::waitKey(1)=='q'){
        break;
    }
    }
    return 0;
}
