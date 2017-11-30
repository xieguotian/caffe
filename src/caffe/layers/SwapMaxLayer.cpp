#include "caffe/layers/SwapMaxLayer.hpp"

namespace caffe{
#ifdef CPU_ONLY
	STUB_GPU(SwapMaxLayer);
#endif

	INSTANTIATE_CLASS(SwapMaxLayer);
	REGISTER_LAYER_CLASS(SwapMax);
}
