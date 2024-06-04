#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

// 包含必要的CUDA头文件
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>
#include "NvInferPlugin.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace nvinfer1;
using namespace cv;
#define GPU_BLOCK_THREADS 1024
#define NUM_BOX_ELEMENT 8
class Myyolo
{
  private:
  void* buffers[2];
  int INPUT_H = 640;
  int INPUT_W = 640;
  int CLASSES = 80;
  int Num_box = 8400;
  float CONF_THRESHOLD = 0.1;
  float NMS_THRESHOLD = 0.5;
  float MASK_THRESHOLD = 0.5;
  int OUTPUT_SIZE = Num_box * (CLASSES+4);//output0
  const char* INPUT_BLOB_NAME = "images";       //这里对应模型的输入和输出名称，通过onnx文件在netron中可查看
  const char* OUTPUT_BLOB_NAME = "output0";
  char* trtModelStream=nullptr;

  uint8_t* src_device = nullptr;   //指向device上的图像位置的指针
  float* affine_matrix_device=nullptr;  //指向device上的仿射变换矩阵的位置的指针
  float* output_array=nullptr;    //指向device上的后处理结果的指针

  public:
  static const int MAX_IMAGE_BOXES=1024;
  float prob[1 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT];  //host上的后处理结果，output_array(device)->prob(host)
  size_t size=0;

  Myyolo(const char* modeltrtpath);
  void preprocess(Mat& src);
  void infer();
  void decode_nms();
  ~Myyolo();
  /*调用cuda api推理所需要的配置*/
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  Logger gLogger;
  cudaStream_t stream;
};
/*预处理时所用图像处理方法*/
enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };
enum class ChannelType : int { None = 0, SwapRB = 1 };

struct Norm {
  float mean[3];
  float std[3];
  float alpha, beta;
  NormType type = NormType::None;
  ChannelType channel_type = ChannelType::None;

  // out = (x * alpha - mean) / std
  static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f,
                       ChannelType channel_type = ChannelType::None);

  // out = x * alpha + beta
  static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

  // None
  static Norm None();
};
/*根据源图像大小和模型要求输入大小确定仿射变换矩阵*/
struct AffineMatrix {
  float i2d[6];  // image to dst(network), 2x3 matrix
  float d2i[6];  // dst to image, 2x3 matrix

  void compute(const std::tuple<int, int> &from, const std::tuple<int, int> &to) {
    float scale_x = get<0>(to) / (float)get<0>(from);
    float scale_y = get<1>(to) / (float)get<1>(from);
    float scale = std::min(scale_x, scale_y);
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = -scale * get<0>(from) * 0.5 + get<0>(to) * 0.5 + scale * 0.5 - 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = -scale * get<1>(from) * 0.5 + get<1>(to) * 0.5 + scale * 0.5 - 0.5;

    double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
    D = D != 0. ? double(1.) / D : double(0.);
    double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
    double b1 = -A11 * i2d[2] - A12 * i2d[5];
    double b2 = -A21 * i2d[2] - A22 * i2d[5];

    d2i[0] = A11;
    d2i[1] = A12;
    d2i[2] = b1;
    d2i[3] = A21;
    d2i[4] = A22;
    d2i[5] = b2;
  }
};
/*图像预处理和后处理时调用CUDA加速对应的核函数*/
void warp_affine_bilinear_and_normalize_plane(uint8_t *src, int src_line_size, int src_width,
                                                     int src_height, float *dst, int dst_width,
                                                     int dst_height, float *matrix_2_3,
                                                     uint8_t const_value, const Norm &norm,
                                                    cudaStream_t stream);
__global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width,
    int dst_height, uint8_t const_value_st, float *warp_affine_matrix_2_3, Norm norm);
static dim3 grid_dims(int numJobs);
static dim3 block_dims(int numJobs);
static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox,
                                               float *oy);
static __global__ void decode_kernel_v8(float *predict, int num_bboxes, int num_classes,
                                        int output_cdim, float confidence_threshold,
                                        float *invert_affine_matrix, float *parray,
                                        int MAX_IMAGE_BOXES);
static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom);
static __global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold);
void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                                  cudaStream_t stream);                              
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

struct Box {
  float left, top, right, bottom, confidence;
  int class_label;
  //std::shared_ptr<InstanceSegmentMap> seg;  // valid only in segment task

  Box() = default;
  Box(float left, float top, float right, float bottom, float confidence, int class_label)
      : left(left),
        top(top),
        right(right),
        bottom(bottom),
        confidence(confidence),
        class_label(class_label) {}
};
#endif