#include <vector>

#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

  template <typename Dtype>
  void BiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top)
  {
    if (bottom.size() == 1 && this->blobs_.size() > 0)
    {
      LOG(INFO) << "Skipping parameter initialization";
    }
    else if (bottom.size() == 1)
    {
      // bias is a learned parameter; initialize it
      const BiasParameter &param = this->layer_param_.bias_param();
      const int axis = bottom[0]->CanonicalAxisIndex(param.axis()); //默认是1
      const int num_axes = param.num_axes();                        //默认是1
      CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                             << "or -1 to extend to the end of bottom[0]";
      if (num_axes >= 0)
      {
        CHECK_GE(bottom[0]->num_axes(), axis + num_axes)
            << "bias blob's shape extends past bottom[0]'s shape when applied "
            << "starting with bottom[0] axis = " << axis;
      }
      this->blobs_.resize(1); //可学习参数beta
      const vector<int>::const_iterator &shape_start =
          bottom[0]->shape().begin() + axis;
      const vector<int>::const_iterator &shape_end =
          (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
      vector<int> bias_shape(shape_start, shape_end);
      this->blobs_[0].reset(new Blob<Dtype>(bias_shape)); //通道数
    }
  }

  template <typename Dtype>
  void BiasLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top)
  {
    const BiasParameter &param = this->layer_param_.bias_param();
    Blob<Dtype> *bias = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
    // Always set axis == 0 in special case where bias is a scalar
    // (num_axes == 0). Mathematically equivalent for any choice of axis, so the
    // actual setting can be safely ignored; and computation is most efficient
    // with axis == 0 and (therefore) outer_dim_ == 1.
    const int axis = (bias->num_axes() == 0) ? 0 : bottom[0]->CanonicalAxisIndex(param.axis());
    CHECK_GE(bottom[0]->num_axes(), axis + bias->num_axes())
        << "bias blob's shape extends past bottom[0]'s shape when applied "
        << "starting with bottom[0] axis = " << axis;
    for (int i = 0; i < bias->num_axes(); ++i)
    {
      CHECK_EQ(bottom[0]->shape(axis + i), bias->shape(i))
          << "dimension mismatch between bottom[0]->shape(" << axis + i
          << ") and bias->shape(" << i << ")";
    }
    outer_dim_ = bottom[0]->count(0, axis);                 //batch size
    bias_dim_ = bias->count();                              //channels
    inner_dim_ = bottom[0]->count(axis + bias->num_axes()); //f_h*f_w
    dim_ = bias_dim_ * inner_dim_;                          //每一个特征图的偏移量
    if (bottom[0] != top[0])
    {
      top[0]->ReshapeLike(*bottom[0]);
    }
    bias_multiplier_.Reshape(vector<int>(1, inner_dim_));
    if (bias_multiplier_.cpu_data()[inner_dim_ - 1] != Dtype(1))
    {
      caffe_set(inner_dim_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }
  }

  template <typename Dtype>
  void BiasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top)
  {
    const Dtype *bias_data =
        ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();
    if (bottom[0] != top[0])
    {
      const Dtype *bottom_data = bottom[0]->cpu_data();
      caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }
    for (int n = 0; n < outer_dim_; ++n)
    {
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bias_dim_,
                     inner_dim_, 1, Dtype(1), bias_data,
                     bias_multiplier_.cpu_data(), Dtype(1), top_data);
      top_data += dim_;
    }
  }

  INSTANTIATE_CLASS(BiasLayer);
  REGISTER_LAYER_CLASS(Bias);

} // namespace caffe
