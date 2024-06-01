#include "test.h"


Norm Norm::mean_std(const float mean[3], const float std[3], float alpha,
                    ChannelType channel_type) {
  Norm out;
  out.type = NormType::MeanStd;
  out.alpha = alpha;
  out.channel_type = channel_type;
  memcpy(out.mean, mean, sizeof(out.mean));
  memcpy(out.std, std, sizeof(out.std));
  return out;
}

Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type) {
  Norm out;
  out.type = NormType::AlphaBeta;
  out.alpha = alpha;
  out.beta = beta;
  out.channel_type = channel_type;
  return out;
}

Norm Norm::None() { return Norm(); }
__global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    uint8_t *src, int src_line_size, int src_width, int src_height, float *dst, int dst_width,
    int dst_height, uint8_t const_value_st, float *warp_affine_matrix_2_3, Norm norm) {
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= dst_width || dy >= dst_height) return;

  float m_x1 = warp_affine_matrix_2_3[0];
  float m_y1 = warp_affine_matrix_2_3[1];
  float m_z1 = warp_affine_matrix_2_3[2];
  float m_x2 = warp_affine_matrix_2_3[3];
  float m_y2 = warp_affine_matrix_2_3[4];
  float m_z2 = warp_affine_matrix_2_3[5];

  float src_x = m_x1 * dx + m_y1 * dy + m_z1;
  float src_y = m_x2 * dx + m_y2 * dy + m_z2;
  float c0, c1, c2;

  if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
    // out of range
    c0 = const_value_st;
    c1 = const_value_st;
    c2 = const_value_st;
  } else {
    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
    float ly = src_y - y_low;
    float lx = src_x - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    uint8_t *v1 = const_value;
    uint8_t *v2 = const_value;
    uint8_t *v3 = const_value;
    uint8_t *v4 = const_value;
    if (y_low >= 0) {
      if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3;

      if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
    }

    if (y_high < src_height) {
      if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;

      if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
    }

    // same to opencv
    c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
    c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
    c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
  }

  if (norm.channel_type == ChannelType::SwapRB) {
    float t = c2;
    c2 = c0;
    c0 = t;
  }

  if (norm.type == NormType::MeanStd) {
    c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
    c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
    c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
  } else if (norm.type == NormType::AlphaBeta) {
    c0 = c0 * norm.alpha + norm.beta;
    c1 = c1 * norm.alpha + norm.beta;
    c2 = c2 * norm.alpha + norm.beta;
  }

  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;

  // float* pdst = dst + dy * 640*3 + dx * 3;
  //   pdst[0] = c0; pdst[1] = c1; pdst[2] = c2;
}

void warp_affine_bilinear_and_normalize_plane(uint8_t *src, int src_line_size, int src_width,
                                                     int src_height, float *dst, int dst_width,
                                                     int dst_height, float *matrix_2_3,
                                                     uint8_t const_value, const Norm &norm,
                                                    cudaStream_t stream) {
  dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
  dim3 block(32, 32);

  warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block,0,stream>>>(
      src, src_line_size, src_width, src_height, dst, dst_width, dst_height, const_value,
      matrix_2_3, norm);
}
static dim3 grid_dims(int numJobs) {
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs) {
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static __host__ __device__ void affine_project(float *matrix, float x, float y, float *ox,
                                               float *oy) {
  *ox = matrix[0] * x + matrix[1] * y + matrix[2];
  *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel_v8(float *predict, int num_bboxes, int num_classes,
                                        int output_cdim, float confidence_threshold,
                                        float *invert_affine_matrix, float *parray,
                                        int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float *class_confidence = pitem + 4;
  float confidence = *class_confidence++;
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width = *pitem++;
  float height = *pitem++;
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;  // 1 = keep, 0 = ignore
  *pout_item++ = position;
}
static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom) {
  float cleft = max(aleft, bleft);
  float ctop = max(atop, btop);
  float cright = min(aright, bright);
  float cbottom = min(abottom, bbottom);

  float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
  if (c_area == 0.0f) return 0.0f;

  float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
  float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
  return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold) {
  int position = (blockDim.x * blockIdx.x + threadIdx.x);
  int count = min((int)*bboxes, MAX_IMAGE_BOXES);
  if (position >= count) return;

  // left, top, right, bottom, confidence, class, keepflag
  float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
  for (int i = 0; i < count; ++i) {
    float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
    if (i == position || pcurrent[5] != pitem[5]) continue;

    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && i < position) continue;

      float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                          pitem[2], pitem[3]);

      if (iou > threshold) {
        pcurrent[6] = 0;  // 1=keep, 0=ignore
        return;
      }
    }
  }
}

void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  float *invert_affine_matrix, float *parray, int MAX_IMAGE_BOXES,
                                  cudaStream_t stream) {
  auto grid = grid_dims(num_bboxes);
  auto block = block_dims(num_bboxes);

  
  decode_kernel_v8<<<grid, block, 0, stream>>>(
      predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
      parray, MAX_IMAGE_BOXES);

  grid = grid_dims(MAX_IMAGE_BOXES);
  block = block_dims(MAX_IMAGE_BOXES);
  fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold);
}

// void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
// {
//     const ICudaEngine& engine = context.getEngine();

//     // Pointers to input and output device buffers to pass to engine.
//     // Engine requires exactly IEngine::getNbBindings() number of buffers.
//     //assert(engine.getNbBindings() == 2);
//     void* buffers[2];

//     // In order to bind the buffers, we need to know the names of the input and output tensors.
//     // Note that indices are guaranteed to be less than IEngine::getNbBindings()
//     const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
// 	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

//     // Create GPU buffers on device
//     CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));//
// 	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
// 	//CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float)));
// 	// cudaMalloc分配内存 cudaFree释放内存 cudaMemcpy或 cudaMemcpyAsync 在主机和设备之间传输数据
// 	// cudaMemcpy cudaMemcpyAsync 显式地阻塞传输 显式地非阻塞传输 
//     // Create stream
//     cudaStream_t stream;
//     CHECK(cudaStreamCreate(&stream));

//     // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//     CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//     context.enqueue(batchSize, buffers, stream, nullptr);
// 	//CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
// 	cudaStreamSynchronize(stream);

//     // Release stream and buffers
//     cudaStreamDestroy(stream);
//     CHECK(cudaFree(buffers[inputIndex]));
// 	CHECK(cudaFree(buffers[outputIndex]));
// 	//CHECK(cudaFree(buffers[outputIndex1]));
// }
