#include "caffe/layers/base_conv_layer.hpp"

namespace caffe
{
    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                 const vector<Blob<Dtype> *> &top)
    {
        // Configure the kernel size, padding, stride, and inputs.
        ConvolutionParameter conv_param = this->layer_param_.convolution_param();
        force_nd_im2col_ = conv_param.force_nd_im2col();                  //默认是false
        channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis()); //channel_axis_ =1
        const int first_spatial_axis = channel_axis_ + 1;                 //first_spatial_axis=2
        const int num_axes = bottom[0]->num_axes();                       //num_axes=4
        num_spatial_axes_ = num_axes - first_spatial_axis;                //num_spatial_axes_=2
        CHECK_GE(num_spatial_axes_, 0);
        vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
        // 设置卷积的shape
        // Setup filter kernel dimensions (kernel_shape_).
        kernel_shape_.Reshape(spatial_dim_blob_shape); //shape=2
        int *kernel_shape_data = kernel_shape_.mutable_cpu_data();
        if (conv_param.has_kernel_h() || conv_param.has_kernel_w())
        {
            //不存在kernel_size这个参数时的操作
            CHECK_EQ(num_spatial_axes_, 2)
                << "kernel_h & kernel_w can only be used for 2D convolution.";
            CHECK_EQ(0, conv_param.kernel_size_size())
                << "Either kernel_size or kernel_h/w should be specified; not both.";
            kernel_shape_data[0] = conv_param.kernel_h();
            kernel_shape_data[1] = conv_param.kernel_w();
        }
        else
        {
            const int num_kernel_dims = conv_param.kernel_size_size();
            CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
                << "kernel_size must be specified once, or once per spatial dimension "
                << "(kernel_size specified " << num_kernel_dims << " times; "
                << num_spatial_axes_ << " spatial dims).";
            for (int i = 0; i < num_spatial_axes_; ++i)
            {
                kernel_shape_data[i] = conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
                CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
            }
        }
        // Setup stride dimensions (stride_).
        stride_.Reshape(spatial_dim_blob_shape);
        int *stride_data = stride_.mutable_cpu_data();
        if (conv_param.has_stride_h() || conv_param.has_stride_w())
        {
            CHECK_EQ(num_spatial_axes_, 2)
                << "stride_h & stride_w can only be used for 2D convolution.";
            CHECK_EQ(0, conv_param.stride_size())
                << "Either stride or stride_h/w should be specified; not both.";
            stride_data[0] = conv_param.stride_h();
            stride_data[1] = conv_param.stride_w();
        }
        else
        {
            const int num_stride_dims = conv_param.stride_size();
            CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
                  num_stride_dims == num_spatial_axes_)
                << "stride must be specified once, or once per spatial dimension "
                << "(stride specified " << num_stride_dims << " times; "
                << num_spatial_axes_ << " spatial dims).";
            const int kDefaultStride = 1;
            for (int i = 0; i < num_spatial_axes_; ++i)
            {
                stride_data[i] = (num_stride_dims == 0) ? kDefaultStride : conv_param.stride((num_stride_dims == 1) ? 0 : i);
                CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
            }
        }
        // Setup pad dimensions (pad_).
        pad_.Reshape(spatial_dim_blob_shape);
        int *pad_data = pad_.mutable_cpu_data();
        if (conv_param.has_pad_h() || conv_param.has_pad_w())
        {
            CHECK_EQ(num_spatial_axes_, 2)
                << "pad_h & pad_w can only be used for 2D convolution.";
            CHECK_EQ(0, conv_param.pad_size())
                << "Either pad or pad_h/w should be specified; not both.";
            pad_data[0] = conv_param.pad_h();
            pad_data[1] = conv_param.pad_w();
        }
        else
        {
            const int num_pad_dims = conv_param.pad_size();
            CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
                  num_pad_dims == num_spatial_axes_)
                << "pad must be specified once, or once per spatial dimension "
                << "(pad specified " << num_pad_dims << " times; "
                << num_spatial_axes_ << " spatial dims).";
            const int kDefaultPad = 0;
            for (int i = 0; i < num_spatial_axes_; ++i)
            {
                pad_data[i] = (num_pad_dims == 0) ? kDefaultPad : conv_param.pad((num_pad_dims == 1) ? 0 : i);
            }
        }
        // Setup dilation dimensions (dilation_).
        dilation_.Reshape(spatial_dim_blob_shape);
        int *dilation_data = dilation_.mutable_cpu_data();
        const int num_dilation_dims = conv_param.dilation_size();
        CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
              num_dilation_dims == num_spatial_axes_)
            << "dilation must be specified once, or once per spatial dimension "
            << "(dilation specified " << num_dilation_dims << " times; "
            << num_spatial_axes_ << " spatial dims).";
        const int kDefaultDilation = 1;
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation : conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
        }
        // Special case: im2col is the identity for 1x1 convolution with stride 1
        // and no padding, so flag for skipping the buffer and transformation.
        is_1x1_ = true;
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            is_1x1_ &=
                kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
            if (!is_1x1_)
            {
                break;
            }
        }
        // Configure output channels and groups.
        channels_ = bottom[0]->shape(channel_axis_);                       //input_channel
        num_output_ = this->layer_param_.convolution_param().num_output(); //output_channel
        CHECK_GT(num_output_, 0);
        group_ = this->layer_param_.convolution_param().group();
        CHECK_EQ(channels_ % group_, 0);
        CHECK_EQ(num_output_ % group_, 0) << "Number of output should be multiples of group.";
        if (reverse_dimensions())
        {
            conv_out_channels_ = channels_;
            conv_in_channels_ = num_output_;
        }
        else
        {
            conv_out_channels_ = num_output_;
            conv_in_channels_ = channels_;
        }
        // Handle the parameters: weights and biases.
        // - blobs_[0] holds the filter weights
        // - blobs_[1] holds the biases (optional)
        vector<int> weight_shape(2);
        weight_shape[0] = conv_out_channels_;
        weight_shape[1] = conv_in_channels_ / group_;
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            weight_shape.push_back(kernel_shape_data[i]);
        } //卷积核的尺寸[output_channel,input_channel,kernel_height,kernel_width]
        bias_term_ = this->layer_param_.convolution_param().bias_term();
        vector<int> bias_shape(bias_term_, num_output_);
        if (this->blobs_.size() > 0) //一般this->blobs_.size()=0，因此还没有初始化其大小
        {
            CHECK_EQ(1 + bias_term_, this->blobs_.size())
                << "Incorrect number of weight blobs.";
            if (weight_shape != this->blobs_[0]->shape())
            {
                Blob<Dtype> weight_shaped_blob(weight_shape);
                LOG(FATAL) << "Incorrect weight shape: expected shape "
                           << weight_shaped_blob.shape_string() << "; instead, shape was "
                           << this->blobs_[0]->shape_string();
            }
            if (bias_term_ && bias_shape != this->blobs_[1]->shape())
            {
                Blob<Dtype> bias_shaped_blob(bias_shape);
                LOG(FATAL) << "Incorrect bias shape: expected shape "
                           << bias_shaped_blob.shape_string() << "; instead, shape was "
                           << this->blobs_[1]->shape_string();
            }
            LOG(INFO) << "Skipping parameter initialization";
        }
        else
        {
            if (bias_term_)
            {
                this->blobs_.resize(2);
            }
            else
            {
                this->blobs_.resize(1); //这个blobs_来源于layer.hpp中
            }
            // Initialize the weights:
            // output channels x input channels per-group x kernel height x kernel width
            this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
            // If necessary, initialize the biases.
            if (bias_term_)
            {
                this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
            }
        }
        kernel_dim_ = this->blobs_[0]->count(1);                    // input_channel*kernel_height*kernel_width
        weight_offset_ = conv_out_channels_ * kernel_dim_ / group_; //权重偏移量 c_out*c_in*k_h*k_w
    }

    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                              const vector<Blob<Dtype> *> &top)
    {
        const int first_spatial_axis = channel_axis_ + 1; //first_spatial_axis=2
        CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
            << "bottom num_axes may not change.";
        num_ = bottom[0]->count(0, channel_axis_); //batch size
        CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
            << "Input size incompatible with convolution kernel.";
        // TODO: generalize to handle inputs of different shapes.
        for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id)
        {
            // 存在多个bottom的情况
            CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
                << "shape mismatch - bottom[0]: " << bottom[0]->shape_string()
                << " vs. bottom[" << bottom_id << "]: "
                << bottom[bottom_id]->shape_string();
        }
        // Shape the tops.
        bottom_shape_ = &bottom[0]->shape(); // N * channels * height * width
        compute_output_shape();              //计算算计后特征图的尺寸，会改变output_shape_的值
        vector<int> top_shape(bottom[0]->shape().begin(),
                              bottom[0]->shape().begin() + channel_axis_);
        top_shape.push_back(num_output_);
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            top_shape.push_back(output_shape_[i]);
        } // top_shape = [batch_size,output_channel,feature_height,feature_width]
        for (int top_id = 0; top_id < top.size(); ++top_id)
        {
            top[top_id]->Reshape(top_shape);
        }
        if (reverse_dimensions())
        {
            conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
        }
        else
        {
            conv_out_spatial_dim_ = top[0]->count(first_spatial_axis); // feature_height * feature_width
        }
        col_offset_ = kernel_dim_ * conv_out_spatial_dim_;                    //TODO c_in*k_h*k_w*f_h*f_w 行偏移量?
        output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_; //输出偏移量 c_out*f_h/f_w
        // Setup input dimensions (conv_input_shape_).
        vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1); //[3]
        conv_input_shape_.Reshape(bottom_dim_blob_shape);
        int *conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
        for (int i = 0; i < num_spatial_axes_ + 1; ++i)
        {
            if (reverse_dimensions())
            {
                conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
            }
            else
            {
                conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i); //input的shape
            }
        }
        // The im2col result buffer will only hold one image at a time to avoid
        // overly large memory usage. In the special case of 1x1 convolution
        // it goes lazily unused to save memory.
        col_buffer_shape_.clear();
        col_buffer_shape_.push_back(kernel_dim_ * group_);
        for (int i = 0; i < num_spatial_axes_; ++i)
        {
            if (reverse_dimensions())
            {
                col_buffer_shape_.push_back(input_shape(i + 1));
            }
            else
            {
                col_buffer_shape_.push_back(output_shape_[i]);
            }
        }
        // 将bottom blob转化为im2col,其对应的shape为[c_in*k_h*k_w,f_h,f_w]
        // 其中f_h,f_w表示输入的特征图的高宽,第一项表示卷积核的数据,同时也是一次卷积操作的数量
        col_buffer_.Reshape(col_buffer_shape_);
        bottom_dim_ = bottom[0]->count(channel_axis_);                   // 表明每个batch的偏移量,总共有num_个batch
        top_dim_ = top[0]->count(channel_axis_);                         // 表明每个batch的偏移量,总共有num_个batch
        num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_; //卷积核的数目
        // Set up the all ones "bias multiplier" for adding biases by BLAS
        // 将"bias multiplier"全部设置为1,为了后面计算bias方便
        // bias的数目是top blob的空间shape
        // TODO 如果不使用bias,是否在这里可省下空间
        out_spatial_dim_ = top[0]->count(first_spatial_axis);
        if (bias_term_)
        {
            vector<int> bias_multiplier_shape(1, out_spatial_dim_);
            bias_multiplier_.Reshape(bias_multiplier_shape);
            caffe_set(bias_multiplier_.count(), Dtype(1),
                      bias_multiplier_.mutable_cpu_data());
        }
    }

    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype *input,
                                                       const Dtype *weights, Dtype *output, bool skip_im2col)
    {
        const Dtype *col_buff = input;
        if (!is_1x1_)
        {
            // 如果不是1*1的操作
            if (!skip_im2col)
            {
                conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
            }
            col_buff = col_buffer_.cpu_data();
        }
        for (int g = 0; g < group_; ++g)
        {
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_,
                                  (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
                                  (Dtype)0., output + output_offset_ * g);
        }
    }

    template <typename Dtype>
    void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype *output,
                                                       const Dtype *bias)
    {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
                              out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
                              (Dtype)1., output);
    }

    INSTANTIATE_CLASS(BaseConvolutionLayer);

} // namespace caffe