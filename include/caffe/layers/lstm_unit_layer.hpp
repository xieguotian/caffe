#ifndef CAFFE_LSTM_UNIT_LAYERS_HPP_
#define CAFFE_LSTM_UNIT_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	/**
	* @brief A helper for LSTMLayer: computes a single timestep of the
	*        non-linearity of the LSTM, producing the updated cell and hidden
	*        states.
	*/
	template <typename Dtype>
	class LSTMUnitLayer : public Layer<Dtype> {
	public:
		explicit LSTMUnitLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "LSTMUnit"; }
		virtual inline int ExactNumBottomBlobs() const { return 3; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

		virtual inline bool AllowForceBackward(const int bottom_index) const {
			// Can't propagate to sequence continuation indicators.
			return bottom_index != 2;
		}

	protected:
		/**
		* @param bottom input Blob vector (length 3)
		*   -# @f$ (1 \times N \times D) @f$
		*      the previous timestep cell state @f$ c_{t-1} @f$
		*   -# @f$ (1 \times N \times 4D) @f$
		*      the "gate inputs" @f$ [i_t', f_t', o_t', g_t'] @f$
		*   -# @f$ (1 \times N) @f$
		*      the sequence continuation indicators  @f$ \delta_t @f$
		* @param top output Blob vector (length 2)
		*   -# @f$ (1 \times N \times D) @f$
		*      the updated cell state @f$ c_t @f$, computed as:
		*          i_t := \sigmoid[i_t']
		*          f_t := \sigmoid[f_t']
		*          o_t := \sigmoid[o_t']
		*          g_t := \tanh[g_t']
		*          c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
		*   -# @f$ (1 \times N \times D) @f$
		*      the updated hidden state @f$ h_t @f$, computed as:
		*          h_t := o_t .* \tanh[c_t]
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		/**
		* @brief Computes the error gradient w.r.t. the LSTMUnit inputs.
		*
		* @param top output Blob vector (length 2), providing the error gradient with
		*        respect to the outputs
		*   -# @f$ (1 \times N \times D) @f$:
		*      containing error gradients @f$ \frac{\partial E}{\partial c_t} @f$
		*      with respect to the updated cell state @f$ c_t @f$
		*   -# @f$ (1 \times N \times D) @f$:
		*      containing error gradients @f$ \frac{\partial E}{\partial h_t} @f$
		*      with respect to the updated cell state @f$ h_t @f$
		* @param propagate_down see Layer::Backward.
		* @param bottom input Blob vector (length 3), into which the error gradients
		*        with respect to the LSTMUnit inputs @f$ c_{t-1} @f$ and the gate
		*        inputs are computed.  Computatation of the error gradients w.r.t.
		*        the sequence indicators is not implemented.
		*   -# @f$ (1 \times N \times D) @f$
		*      the error gradient w.r.t. the previous timestep cell state
		*      @f$ c_{t-1} @f$
		*   -# @f$ (1 \times N \times 4D) @f$
		*      the error gradient w.r.t. the "gate inputs"
		*      @f$ [
		*          \frac{\partial E}{\partial i_t}
		*          \frac{\partial E}{\partial f_t}
		*          \frac{\partial E}{\partial o_t}
		*          \frac{\partial E}{\partial g_t}
		*          ] @f$
		*   -# @f$ (1 \times 1 \times N) @f$
		*      the gradient w.r.t. the sequence continuation indicators
		*      @f$ \delta_t @f$ is currently not computed.
		*/
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		/// @brief The hidden and output dimension.
		int hidden_dim_;
		Blob<Dtype> X_acts_;
	};
}
#endif