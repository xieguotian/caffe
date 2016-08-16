#ifndef HALF_UTIL_HPP
#define HALF_UTIL_HPP
//#include "THCGeneral.h"
#define CUDA_HAS_FP16 1
/* We compile with CudaHalfTensor support if we have this: */
#if CUDA_VERSION >= 7050 || CUDA_HAS_FP16
#define CUDA_HALF_TENSOR 1
#endif

/* Kernel side: Native fp16 ALU instructions are available if we have this: */
#if defined(CUDA_HALF_TENSOR) && (__CUDA_ARCH__ >= 530)
#define CUDA_HALF_INSTRUCTIONS 1
#endif

#ifdef CUDA_HALF_TENSOR

#include <cuda_fp16.h>
#include <stdint.h>

namespace caffe{
	template <typename Dtype>
	void THCFloat2Half(/*THCState *state,*/ half *out, Dtype *in, long len);
	template <typename Dtype>
	void THCHalf2Float(/*THCState *state,*/ Dtype *out, half *in, long len);
	half THC_float2half(float a);
	float THC_half2float(half a);
}

/* Check for native fp16 support on the current device (CC 5.3+) */
//int THC_nativeHalfInstructions(THCState *state);

#endif /* CUDA_HALF_TENSOR */

#endif