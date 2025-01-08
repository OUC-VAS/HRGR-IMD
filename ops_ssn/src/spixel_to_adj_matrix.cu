#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/nn/functional.h>


#define CUDA_KERNEL_LOOP(i, n)                                   \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256; // TODO


/**
 * [1, CUDA_NUM_THREADS] -> 1;
 * [1 + CUDA_NUM_THREADS, CUDA_NUM_THREADS * 2] -> 2
 * @param N 总共的 kernel 数
 * @return kernel 可以被分为多少个 block
 */
inline int GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

namespace F = torch::nn::functional;

template<typename scalar_t>
__global__ void
spixel2d_forward_kernel(const int n, const scalar_t *data_input, int *data_output, const int height, const int width,
                        const int h_pad, const int w_pad, const int n_pixel) {
    // n 代表了总共一个 step 的 kernel 的个数，与 input 的元素个数是相同的，index 代表了第几个 kernel。
    const int height_padded = height + h_pad * 2;
    const int width_padded = height + w_pad * 2;
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int w = _temp % width;
        _temp /= width;
        const int h = _temp % height;
        _temp /= height;
        const int b = _temp;
        const scalar_t *data_input_ptr = data_input + b * height_padded * width_padded;
        int *data_output_ptr = data_output + b * n_pixel * n_pixel;
        const int _h = h + h_pad;
        const int _w = w + w_pad;
        const int center_label = data_input_ptr[_h * width_padded + _w];
        for (int i = -h_pad; i <= h_pad; i++) {
            for (int j = -w_pad; j <= w_pad; j++) {
                const int shift_label = data_input_ptr[(_h + i) * width_padded + (_w + j)];
                data_output_ptr[center_label * n_pixel + shift_label] = 1;
                data_output_ptr[shift_label * n_pixel + center_label] = 1;
//                std::printf("[*data_output_ptr: %d\n, (center-shift: %d, shift-center: %d)]", *data_output_ptr,
//                            data_output_ptr[center_label * n_pixel + shift_label],
//                            data_output_ptr[shift_label * n_pixel + center_label]);
//                std::printf("[index: %d][b: %d, h: %d, w: %d][i: %d, j: %d][center: %d, shift: %d]\n", index, b, h, w,
//                            i, j, center_label, shift_label);

//                atomicExch(data_output_ptr + center_label * n_pixel + shift_label, 1);
//                atomicExch(data_output_ptr + shift_label * n_pixel + center_label, 1);
            }
        }
    }
}

at::Tensor spixel2d_forward_cuda(const at::Tensor &input, const int n_pixel) {
    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERT(input.dim() == 3, "input dim must be 3: (B, H, W)");

    const int batch_size = input.size(0);
    const int height = input.size(1);
    const int width = input.size(2);
    const int h_pad = 1;
    const int w_pad = 1;
    auto padded = F::pad(input, F::PadFuncOptions({h_pad, h_pad, w_pad, w_pad}).mode(torch::kReplicate));

    auto output = at::zeros({batch_size, n_pixel, n_pixel}, at::dtype<int>().device(input.device()));
    AT_DISPATCH_FLOATING_TYPES(input.type(), "spixel_to_adj_matrix_forward", ([&] {
        // 转化为 1 维数组，方便传入 GPU
        const scalar_t *data_padded = padded.data<scalar_t>();
        int *data_output = output.data<int>();
        int num_kernels = batch_size * height * width;
        spixel2d_forward_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(num_kernels, data_padded, data_output,
                                                                               height, width, h_pad, w_pad, n_pixel);
    }));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in spixel_to_adj_matrix_forward: %s\n", cudaGetErrorString(err));
    }
    return output;
}


template<typename scalar_t>
__global__ void
spixel3d_forward_kernel(const int n, const scalar_t *data_input, int *data_output, const int depth, const int height,
                        const int width, const int d_pad, const int h_pad, const int w_pad, const int n_pixel) {
    // n 代表了总共一个 step 的 kernel 的个数，与 input 的元素个数是相同的，index 代表了第几个 kernel。
    const int height_padded = height + h_pad * 2;
    const int width_padded = height + w_pad * 2;
    const int depth_padded = depth + d_pad * 2;
    CUDA_KERNEL_LOOP(index, n) {
        int _temp = index;
        const int w = _temp % width;
        _temp /= width;
        const int h = _temp % height;
        _temp /= height;
        const int d = _temp % depth;
        _temp /= depth;
        const int b = _temp;
        const scalar_t *data_input_ptr = data_input + b * depth_padded * height_padded * width_padded;
        int *data_output_ptr = data_output + b * n_pixel * n_pixel;
        const int _d = d + d_pad;
        const int _h = h + h_pad;
        const int _w = w + w_pad;
        const int center_label = data_input_ptr[(_d * height_padded + _h) * width_padded + _w];
        for (int k = -d_pad; k <= d_pad; k++) {
            for (int i = -h_pad; i <= h_pad; i++) {
                for (int j = -w_pad; j <= w_pad; j++) {
                    const int shift_label = data_input_ptr[
                            ((_d + k) * height_padded + (_h + i)) * width_padded + (_w + j)];
//                    std::printf("[index: %d][d: %d, h: %d, w: %d][k: %d, i: %d, j: %d][center: %d, shift: %d]\n", index,
//                                d, h, w, k, i, j, center_label, shift_label);
                    data_output_ptr[center_label * n_pixel + shift_label] = 1;
                    data_output_ptr[shift_label * n_pixel + center_label] = 1;
                }
            }
        }
    }
}

at::Tensor spixel3d_forward_cuda(const at::Tensor &input, const int n_pixel) {
    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERT(input.dim() == 4, "input dim must be 4: (B, D, H, W)");

    const int batch_size = input.size(0);
    const int depth = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);
    const int d_pad = 1;
    const int h_pad = 1;
    const int w_pad = 1;
    auto padded = F::pad(input, F::PadFuncOptions({d_pad, d_pad, h_pad, h_pad, w_pad, w_pad}).
            mode(torch::kReplicate));
    auto output = at::zeros({batch_size, n_pixel, n_pixel}, at::dtype<int>().device(input.device()));
    AT_DISPATCH_FLOATING_TYPES(input.type(), "spixel_to_adj_matrix_forward", ([&] {
        // 转化为 1 维数组，方便传入 GPU
        const scalar_t *data_padded = padded.data<scalar_t>();
        int *data_output = output.data<int>();
        int num_kernels = batch_size * depth * height * width;
        spixel3d_forward_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(num_kernels, data_padded, data_output,
                                                                               depth, height, width,
                                                                               d_pad, h_pad, w_pad, n_pixel);
    }));
//    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("error in spixel_to_adj_matrix_forward: %s\n", cudaGetErrorString(err));
    }
    return output;
}


//#define TORCH_EXTENSION_NAME spixel_to_adj_matrix_cuda
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("spixel2d_forward", &spixel2d_forward_cuda, "superpixel 2d to adjacency matrix forward (CUDA)");
   m.def("spixel3d_forward", &spixel3d_forward_cuda, "superpixel 3d to adjacency matrix forward (CUDA)");
}
