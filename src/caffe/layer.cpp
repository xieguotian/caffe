#include <boost/thread.hpp>
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

template <typename Dtype>
void Layer<Dtype>::set_field_size(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	if (bottom.size() == 0)
	{
		for (int i = 0; i < top.size(); ++i)
		{
			top[i]->pad = 0;
			top[i]->stride = 1;
			top[i]->kernel_size = 1;
		}
	}
	else
	{

		for (int i = 0; i < top.size(); ++i)
		{
			bool not_initial = true;
			for (int n = 0; n < bottom.size(); n++)
			{
				if (bottom[n] == top[i])
					not_initial = false;
			}
			if (not_initial)
			{
				top[i]->kernel_size = 1;
				top[i]->stride = 1;
				top[i]->pad = 0;
			}
			for (int n = 0; n < bottom.size(); n++)
			{
				if (bottom[n] != top[i])
				{
					if (bottom[n]->kernel_size >= top[i]->kernel_size)
					{
						top[i]->kernel_size = bottom[n]->kernel_size;
						top[i]->pad = bottom[n]->pad;
						top[i]->stride = bottom[n]->stride;
					}
				}
			}
		}
	}
}
INSTANTIATE_CLASS(Layer);

}  // namespace caffe
