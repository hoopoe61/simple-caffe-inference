#include <algorithm>
#include <vector>

#include "caffe/layer_factory.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

    template <typename Dtype>
    void ScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top)
    {
        const ScaleParameter &param = this->layer_param_.scale_param();
        if (bottom.size() == 1 && this->blobs_.size() > 0)
        {
            LOG(INFO) << "Skipping parameter initialization";
        }
        else if (bottom.size() == 1)
        {
            // scale is a learned parameter; initialize it
            axis_ = bottom[0]->CanonicalAxisIndex(param.axis()); //默认是1
            const int num_axes = param.num_axes();               //默认是1
            CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                                   << "or -1 to extend to the end of bottom[0]";
            if (num_axes >= 0)
            {
                CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
                    << "scale blob's shape extends past bottom[0]'s shape when applied "
                    << "starting with bottom[0] axis = " << axis_;
            }
            this->blobs_.resize(1); //可学习参数alpha
            const vector<int>::const_iterator &shape_start =
                bottom[0]->shape().begin() + axis_;
            const vector<int>::const_iterator &shape_end =
                (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
            vector<int> scale_shape(shape_start, shape_end);
            this->blobs_[0].reset(new Blob<Dtype>(scale_shape)); //在BN层后,等同于通道数
        }
        if (param.bias_term())
        {
            LayerParameter layer_param(this->layer_param_);
            layer_param.set_type("Bias");
            BiasParameter *bias_param = layer_param.mutable_bias_param();
            bias_param->set_axis(param.axis());
            if (bottom.size() > 1)
            {
                bias_param->set_num_axes(bottom[1]->num_axes());
            }
            else
            {
                bias_param->set_num_axes(param.num_axes());
            }
            bias_param->mutable_filler()->CopyFrom(param.bias_filler());
            bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
            bias_bottom_vec_.resize(1);
            bias_bottom_vec_[0] = bottom[0];
            bias_layer_->SetUp(bias_bottom_vec_, top);
            if (this->blobs_.size() + bottom.size() < 3)
            {
                // case: blobs.size == 1 && bottom.size == 1
                // or blobs.size == 0 && bottom.size == 2
                bias_param_id_ = this->blobs_.size();
                this->blobs_.resize(bias_param_id_ + 1);
                this->blobs_[bias_param_id_] = bias_layer_->blobs()[0]; //可学习参数beta
            }
            else
            {
                // bias param already initialized
                bias_param_id_ = this->blobs_.size() - 1;
                bias_layer_->blobs()[0] = this->blobs_[bias_param_id_];
            }
        }
    }

    template <typename Dtype>
    void ScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top)
    {
        const ScaleParameter &param = this->layer_param_.scale_param();
        Blob<Dtype> *scale = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
        // Always set axis_ == 0 in special case where scale is a scalar
        // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
        // actual setting can be safely ignored; and computation is most efficient
        // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
        // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
        // performant.)
        axis_ = (scale->num_axes() == 0) ? 0 : bottom[0]->CanonicalAxisIndex(param.axis());
        CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
            << "scale blob's shape extends past bottom[0]'s shape when applied "
            << "starting with bottom[0] axis = " << axis_;
        for (int i = 0; i < scale->num_axes(); ++i)
        {
            CHECK_EQ(bottom[0]->shape(axis_ + i), scale->shape(i))
                << "dimension mismatch between bottom[0]->shape(" << axis_ + i
                << ") and scale->shape(" << i << ")";
        }
        outer_dim_ = bottom[0]->count(0, axis_);
        scale_dim_ = scale->count();
        inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
        top[0]->ReshapeLike(*bottom[0]);
        sum_result_.Reshape(vector<int>(1, outer_dim_ * scale_dim_));
        const int sum_mult_size = std::max(outer_dim_, inner_dim_);
        sum_multiplier_.Reshape(vector<int>(1, sum_mult_size));
        if (sum_multiplier_.cpu_data()[sum_mult_size - 1] != Dtype(1))
        {
            caffe_set(sum_mult_size, Dtype(1), sum_multiplier_.mutable_cpu_data());
        }
        if (bias_layer_)
        {
            bias_bottom_vec_[0] = top[0];
            bias_layer_->Reshape(bias_bottom_vec_, top);
        }
    }

    template <typename Dtype>
    void ScaleLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top)
    {
        const Dtype *bottom_data = bottom[0]->cpu_data();
        const Dtype *scale_data =
            ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
        Dtype *top_data = top[0]->mutable_cpu_data();
        for (int n = 0; n < outer_dim_; ++n)
        {
            for (int d = 0; d < scale_dim_; ++d)
            {
                const Dtype factor = scale_data[d];
                caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
                bottom_data += inner_dim_;
                top_data += inner_dim_;
            }
        }
        if (bias_layer_)
        {
            bias_layer_->Forward(bias_bottom_vec_, top);
        }
    }

    INSTANTIATE_CLASS(ScaleLayer);
    REGISTER_LAYER_CLASS(Scale);

} // namespace caffe
