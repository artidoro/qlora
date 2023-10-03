#include <torch/extension.h>

#include <vector>

#include <c10/cuda/CUDAGuard.h>

// CUDA forward declarations

torch::Tensor bitmask_cuda(
    torch::Tensor tensor, int bit_mask, int workload);

torch::Tensor matmul_cuda_alphas(
    torch::Tensor input,
    torch::Tensor weights, int bit_mask_man, int exp_bits, int chunk_size, int exp_bias, int amode);

torch::Tensor matmul_cuda(
    torch::Tensor input,
    torch::Tensor weights, int bit_mask_man, int exp_bits, int chunk_size, int exp_bias);

//    torch::Tensor weights,  torch::Tensor weights output, int bit_mask_man, int exp_bits, int chunk_size);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



torch::Tensor matmul_alphas(
    torch::Tensor input,
    torch::Tensor weights, int bit_mask_man, int exp_bits, int chunk_size, int exp_bias, int amode){
  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  return matmul_cuda_alphas(input, weights, bit_mask_man, exp_bits, chunk_size, exp_bias, amode);
}

torch::Tensor matmul(
    const torch::Tensor input,
    const torch::Tensor weights, int bit_mask_man, int exp_bits, int chunk_size, int exp_bias){
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  
  // new_shape = torch.Size([input.shape[0], input.shape[1], weights.shape[0]])
  // output = torch.zeros(new_shape, dtype=torch.float32, device=input.device);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  return matmul_cuda(input, weights, bit_mask_man, exp_bits, chunk_size, exp_bias);
}


torch::Tensor bitmask(torch::Tensor tensor, int bit_mask, int workload) {
  
  CHECK_INPUT(tensor);

  return bitmask_cuda( tensor, bit_mask, workload);
}

// torch::Tensor bitmask_round(torch::Tensor tensor, int bit_mask, int workload) {
  
//   CHECK_INPUT(tensor);

//   return bitmask_round( tensor, bit_mask, workload);
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bitmask", &bitmask, "bitmask");
  m.def("matmul_alphas", &matmul_alphas, "matmul_alphas (CUDA)");
  m.def("matmul", &matmul, "matmul (CUDA)");
}