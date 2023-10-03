#include <vector>
#include <torch/extension.h>
#include <assert.h>

#include <math.h>
#include <limits>

#include <algorithm> // std::min
#include <c10/cuda/CUDAStream.h>


__device__ float cuda_quantize(float value, const int bit_mask_man, const float clamp_c, const float clamp_m)
{
    // reineterpret value as int and apply masking, then reinterepret as float:

    int* pvalue = (int*) &value;

    if (value == 0.0){
        return 0.0;
    }

    //float v1 = value;

    if ( value > 0){
        if (value > clamp_c){
            value = clamp_c;
        } else if (value < clamp_m){
            return 0.0;
        }
    } else {
        if (value < -clamp_c){
            value = -clamp_c;
        } else if (value > -clamp_m){
            return 0.0;
        }
    }

    //float v2 = value;

    (*pvalue) = (*pvalue) & bit_mask_man;



    //float v3 = value;

    //printf("v1 = %f, v2 = %f, v3 = %f, clamp_c = %f, clamp_m= %f, mask = %d\n", v1, v2, v3, clamp_c, clamp_m, bit_mask_man);
    return value;
}


__device__ float cuda_quantize_alpha(float value, const int bit_mask_man, 
                                     const float clamp_c, const float clamp_m, 
                                     bool* palpha, const int mode)
{
    int* pvalue = (int*) &value;

    *palpha = true;

    if (value == 0.0){
        return 0.0;
    }

    if ( value > 0){
        if (value > clamp_c){
            value = clamp_c;
            *palpha = (mode == 2);
        } else if (value < clamp_m){
            *palpha = (mode == 1);
            return 0.0;
        }
    } else {
        if (value < -clamp_c){
            *palpha = (mode == 2);
            value = -clamp_c;
        } else if (value > -clamp_m){
            *palpha = (mode == 1);
            return 0.0;
        }
    }

    (*pvalue) = (*pvalue) & bit_mask_man;
    return value;
}


template <typename scalar_t>
__global__ void matmul_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
    scalar_t* __restrict__ output,   //    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    const int chunk_size, const int bit_mask_man, const float clamp_prod_OF, const float clamp_prod_UF, const float clamp_acc_OF, const float clamp_acc_UF,
    const int b1, const int b2, const int d0, const int d1) {



    __shared__ scalar_t reduction_vec[1024];

    // const int n = blockIdx.y;
    // column index
    // const int c = blockIdx.x * blockDim.x + threadIdx.x;


    // const dim3 threads( threads_x , chunks);
    // const dim3 blocks( (d1 +threads_x -1 )/threads_x  , batch_size);


    const int b = blockIdx.y;
    const int d1_ = blockIdx.x * blockDim.x  + threadIdx.x;


    auto b1_ = b / b2;
    auto b2_ = b % b2;

    const int chunk_id = threadIdx.y;
    const int chunks = blockDim.y;

    float epsilon = 0.0000001;

    int max_to_calc = (chunk_id+1)*chunk_size;
    if (max_to_calc > d0){
        max_to_calc = d0;
    }

    float accumulator = 0;
    //int* paccumulator = (int*) &accumulator ;
    //int* pprod = (int*) &prod ;

    // if (b == 0 && chunk_id*chunk_size < max_channel_to_calc)
    //     printf("Cout = %d, h = %d, w = %d, b = %d, chunk_id = %d\n", Cout, h, w, b, chunk_id);

    //printf("(%d,%d) , (%d:%d,%d)\n", b1_, b2_, chunk_id*chunk_size, max_to_calc , d1_);

    if (d1_ < d1){
        for (int d0_ = chunk_id*chunk_size ; d0_ < max_to_calc ; ++d0_){
            float prod = input[b1_][b2_][d0_] * weights[d1_][d0_];
            const float prod_org = prod;

            // if (prod > clamp_c){
            //     prod= clamp_c;
            // } else if (prod < -clamp_c){
            //     prod= -clamp_c;
            // }

            // (*pprod) = (*pprod) & bit_mask_man;

            prod = cuda_quantize(prod, bit_mask_man, clamp_prod_OF, clamp_prod_UF);

            const float accumulator_org = accumulator;
            accumulator = accumulator + prod;

            // if (accumulator > clamp_c){
            //     accumulator = clamp_c;
            // } else if (accumulator < -clamp_c){
            //     accumulator = -clamp_c;
            // }
            
            // (*paccumulator) = (*paccumulator) & bit_mask_man;

            accumulator = cuda_quantize(accumulator, bit_mask_man, clamp_acc_OF, clamp_acc_UF);
            //{b1, b2, d0, d1}

            

            //printf("a = %f, d0= %d, out_idx = %d\n", alpha, d0_, out_idx);
        }
        int out_idx =  (( (b1_)  *b2 + b2_)  *d1 +  d1_);
                                
        if (chunks > 1){

            const int loc = chunk_id + threadIdx.x*chunks;

            reduction_vec[loc] = accumulator;

            __syncthreads();

            if (chunk_id == 0){
                for (int i = 1; i < chunks; ++i){
                    const float accumulator_org = accumulator;
                    const float addon_org = reduction_vec[ threadIdx.x*chunks + i];
                    accumulator += addon_org;

                    // if (accumulator > clamp_c){
                    //     accumulator = clamp_c;
                    // } else if (accumulator < -clamp_c){
                    //     accumulator = -clamp_c;
                    // }

                    // (*paccumulator) = (*paccumulator) & bit_mask_man;
                    accumulator = cuda_quantize(accumulator, bit_mask_man, clamp_acc_OF, clamp_acc_UF);


                }
                output[out_idx] = accumulator;
            }

        } else{
            output[out_idx] = accumulator;
        }
        
    }
}



template <typename scalar_t>
__global__ void matmul_cuda_alphas_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weights,
    bool* __restrict__ output,   //    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> output,
    const int chunk_size, const int bit_mask_man, const float clamp_c, const float clamp_m,
    const int b1, const int b2, const int d0, const int d1, const int mode) {



    __shared__ scalar_t reduction_vec[1024];

    __shared__ bool betas[1024];

    // const int n = blockIdx.y;
    // column index
    // const int c = blockIdx.x * blockDim.x + threadIdx.x;


    // const dim3 threads( threads_x , chunks);
    // const dim3 blocks( (d1 +threads_x -1 )/threads_x  , batch_size);


    const int b = blockIdx.y;
    const int d1_ = blockIdx.x * blockDim.x  + threadIdx.x;

    auto b1_ = b / b2;
    auto b2_ = b % b2;

    const int chunk_id = threadIdx.y;
    const int chunks = blockDim.y;

    float epsilon = 0.0000001;

    int max_to_calc = (chunk_id+1)*chunk_size;
    if (max_to_calc > d0){
        max_to_calc = d0;
    }

    float accumulator = 0;
    float prod = 0;
    //int* paccumulator = (int*) &accumulator ;
    //int* pprod = (int*) &prod ;

    // if (b == 0 && chunk_id*chunk_size < max_channel_to_calc)
    //     printf("Cout = %d, h = %d, w = %d, b = %d, chunk_id = %d\n", Cout, h, w, b, chunk_id);

    //printf("(%d,%d) , (%d:%d,%d)\n", b1_, b2_, chunk_id*chunk_size, max_to_calc , d1_);

    if (d1_ < d1){

        int out_idx_base = ((( (b1_)  *b2 + b2_)*(d1)  + d1_)*d0);
        for (int d0_ = chunk_id*chunk_size ; d0_ < max_to_calc ; ++d0_){
            float prod = input[b1_][b2_][d0_] * weights[d1_][d0_];
            const float prod_org = prod;

            // if (prod > clamp_c){
            //     prod= clamp_c;
            // } else if (prod < -clamp_c){
            //     prod= -clamp_c;
            // }

            // (*pprod) = (*pprod) & bit_mask_man;
            prod = cuda_quantize(prod, bit_mask_man, clamp_c, clamp_m);

            const float accumulator_org = accumulator;
            accumulator = accumulator + prod;

            // if (accumulator > clamp_c){
            //     accumulator = clamp_c;
            // } else if (accumulator < -clamp_c){
            //     accumulator = -clamp_c;
            // }

            // (*paccumulator) = (*paccumulator) & bit_mask_man;
            accumulator = cuda_quantize_alpha(accumulator, bit_mask_man, clamp_c, 
                                              clamp_m, &output[out_idx_base + d0_], mode);
            
            //{b1, b2, d0, d1}

            if (mode == 0){
                const float alpha = (accumulator - accumulator_org)/(prod_org + (prod_org > 0 ? epsilon : -epsilon));
                //printf("a = %f, d0= %d, out_idx = %d\n", alpha, d0_, out_idx);
                //u_int8_t alpha_u8 =  alpha > 2 ? 127 : static_cast<u_int8_t> (alpha*128);
                const bool alpha_u8 =  alpha > 0.0;
                output[out_idx_base + d0_] = alpha_u8;
            }

        }

        int out_idx =  (( (b1_)  *b2 + b2_)  *d1 +  d1_);
                                
        if (chunks > 1){

            const int loc = chunk_id + threadIdx.x*chunks;

            reduction_vec[loc] = accumulator;

            __syncthreads();

            if (chunk_id == 0){

                betas[loc] = 1;

                for (int i = 1; i < chunks; ++i){
                    const float accumulator_org = accumulator;
                    const float addon_org = reduction_vec[ threadIdx.x*chunks + i];
                    accumulator += addon_org;

                    // if (accumulator > clamp_c){
                    //     accumulator = clamp_c;
                    // } else if (accumulator < -clamp_c){
                    //     accumulator = -clamp_c;
                    // }

                    // (*paccumulator) = (*paccumulator) & bit_mask_man;
                    accumulator = cuda_quantize_alpha(accumulator, bit_mask_man, clamp_c, clamp_m, &betas[threadIdx.x*chunks + i], mode);
                    if (mode == 0){
                        float beta = (accumulator - accumulator_org)/(addon_org + (addon_org > 0 ? epsilon : -epsilon));
                        //u_int8_t beta_u8 =  beta > 2 ? 127 : static_cast<u_int8_t> (beta*128);
                        bool beta_u8 =  beta > 0.0;
                        betas[threadIdx.x*chunks + i] = beta_u8;
                    }
                }
            }


            __syncthreads();

            //float beta = betas[loc]/128.0;
            bool beta = betas[loc];
            for (int d0_ = chunk_id*chunk_size ; d0_ < max_to_calc ; ++d0_){
                const int out_idx = out_idx_base +  d0_ ;
                //output[out_idx] = static_cast<u_int8_t> (output[out_idx]*(beta));
                output[out_idx] = output[out_idx] && beta;
                //output[out_idx] = beta;

            }

        } 

    }
}

template <typename scalar_t>
__global__ void bitmask_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tensor,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tensor_out,
    int bit_mask, int workload, int maxwork) {



    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    float value;
    int* pvalue = (int*) &value;
    int max_num = (j+1)*workload;
    if (max_num > maxwork){
        max_num = maxwork;
    }

    for (int k = j*workload; k < max_num; ++k){
        value =  tensor[i][k];
        (*pvalue) = (*pvalue) & bit_mask;
        tensor_out[i][k] = value;
    }

}

torch::Tensor bitmask_cuda(
    torch::Tensor tensor, int bit_mask=-1, int workload = 1){

  const auto x = tensor.size(0);
  const auto y = tensor.size(1);

  auto tensor_out = torch::zeros({x,y}, tensor.device());

  const int threads = 1024;



  const dim3 blocks((x + threads - 1) / (threads), y/workload);
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor.type(), "bitmask_cuda_kernel", ([&] {
    bitmask_cuda_kernel<scalar_t><<<blocks, threads>>>(
        tensor.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        tensor_out.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        bit_mask,workload , y);
  }));


  return tensor_out;
}

torch::Tensor matmul_cuda(
    const torch::Tensor input,
    const torch::Tensor weights,
    int bit_mask_man, int exp_bits, int chunk_size = 16, int exp_bias = 0){
    bool uf = true;
    float clamp_prod_OF;
    float clamp_prod_UF;
    float clamp_acc_OF;
    float clamp_acc_UF;
    int exp_bias_prod = exp_bias-2;
    int exp_bias_acc = exp_bias;

    if (exp_bits < 0) {
        clamp_prod_OF = std::numeric_limits<float>::max();
        clamp_acc_OF = std::numeric_limits<float>::max();
        clamp_prod_UF = 0.0;
        clamp_acc_UF = 0.0;
    }else {
        uint32_t maxm = 1073741823;
        maxm = maxm & bit_mask_man;

        float maxnf = *((float*) &maxm );
        int mpower = (1 << (exp_bits - 1));
        if (exp_bits == 0){
            clamp_prod_OF = maxnf;
            clamp_prod_UF = 1;
            clamp_acc_OF = maxnf;
            clamp_acc_UF = 1;
        }else{
            clamp_prod_OF = maxnf * (1 <<  (mpower+exp_bias_prod-1));
            clamp_acc_OF = maxnf * (1 <<  (mpower+exp_bias_acc-1));

            if (uf){
                clamp_prod_UF =  powf(2.0,  exp_bias_prod - mpower);
                clamp_acc_UF =  powf(2.0,  exp_bias_acc - mpower);
            } else {
                clamp_prod_UF = 0.0;
                clamp_acc_UF = 0.0;
            }

        }
    }


    auto b1 = input.size(0);
    auto b2 = input.size(1);
    auto batch_size = b1*b2;

    const auto d0 = input.size(2);

    const auto d0_b = weights.size(1);
    const auto d1 = weights.size(0);

    assert(d0 == d0_b);

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::zeros({b1, b2, d1}, options );


    const int chunks = (d0 + chunk_size - 1) / chunk_size;
    assert(chunks <= 1024);

    
    int threads_x = (1024 / chunks);
    // clamp at d1:

    if (threads_x > d1){
        threads_x = d1;
    }

    const dim3 threads( threads_x , chunks);


    const dim3 blocks( (d1 + threads_x - 1 )/threads_x , batch_size);

    // printf("threads = %d, %d, %d\n", threads.x, threads.y, threads.z);
    // printf("blocks = %d, %d, %d\n", blocks.x, blocks.y, blocks.z);

    //debug print with all args:

    auto stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "matmul_cuda_kernel", ([&] {
    matmul_cuda_kernel<scalar_t><<<blocks, threads,0, stream>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output.data<scalar_t>(), //output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
        chunk_size, bit_mask_man, clamp_prod_OF, clamp_prod_UF, clamp_acc_OF, clamp_acc_UF,
        b1, b2, d0, d1);
    }));

    return output;
}


torch::Tensor matmul_cuda_alphas(
    torch::Tensor input,
    torch::Tensor weights,
    int bit_mask_man, int exp_bits, int chunk_size = 16, int exp_bias = 0, int amode = 0){
    
    float clamp_c;
    float clamp_m;
    if (exp_bits < 0) {
        clamp_c = std::numeric_limits<float>::max();
    }else {
        uint32_t maxm = 1073741823;
        maxm = maxm & bit_mask_man;

        float maxnf = *((float*) &maxm );
        int mpower = (1 << (exp_bits - 1));
        if (exp_bits == 0){
            clamp_c = maxnf;
            clamp_m = 1;
        }else{
            clamp_c = maxnf * (1 <<  (mpower+exp_bias-1));
            clamp_m = 1.0/ (1 <<  (mpower-exp_bias));
        }
    }


    auto b1 = input.size(0);
    auto b2 = input.size(1);
    auto batch_size = b1*b2;

    const auto d0 = input.size(2);

    const auto d0_b = weights.size(1);
    const auto d1 = weights.size(0);

    assert(d0 == d0_b);

    auto options = torch::TensorOptions().dtype(torch::kBool).device(input.device());
    auto output = torch::zeros({b1, b2, d1, d0}, options );


    const int chunks = (d0 + chunk_size - 1) / chunk_size;
    assert(chunks <= 1024);

    
    int threads_x = (1024 / chunks);

    // clamp at d1:

    if (threads_x > d1){
        threads_x = d1;
    }

    const dim3 threads( threads_x , chunks);


    const dim3 blocks( (d1 + threads_x - 1 )/threads_x , batch_size);

    auto stream = c10::cuda::getCurrentCUDAStream();


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "matmul_cuda_alphas_kernel", ([&] {
    matmul_cuda_alphas_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        input.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        output.data<bool>(), //output.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>()
        chunk_size, bit_mask_man, clamp_c, clamp_m,
        b1, b2, d0, d1, amode);
    }));

    return output;
}
