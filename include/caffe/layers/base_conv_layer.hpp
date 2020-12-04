#ifndef CAFFE_BASE_CONVOLUTION_LAYER_HPP_
#define CAFFE_BASE_CONVOLUTION_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe
{
    /**
   * @brief Abstract base class that factors out the BLAS code common to
   *        ConvolutionLayer and DeconvolutionLayer.
   */
    template <typename Dtype>
    class BaseConvolutionLayer : public Layer<Dtype>
    {
    public:
        explicit BaseConvolutionLayer(const LayerParameter &param)
            : Layer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);
        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline int MinBottomBlobs() const { return 1; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline bool EqualNumBottomTopBlobs() const { return true; }

    protected:
        // Helper functions that abstract away the column buffer and gemm arguments.
        // The last argument in forward_cpu_gemm is so that we can skip the im2col if
        // we just called weight_cpu_gemm with the same input.
        void forward_cpu_gemm(const Dtype *input, const Dtype *weights,
                              Dtype *output, bool skip_im2col = false);
        void forward_cpu_bias(Dtype *output, const Dtype *bias);
#ifndef CPU_ONLY
        void forward_gpu_gemm(const Dtype *col_input, const Dtype *weights,
                              Dtype *output, bool skip_im2col = false);
        void forward_gpu_bias(Dtype *output, const Dtype *bias);
#endif //CPU_ONLY

        /// @brief The spatial dimensions of the input.
        inline int input_shape(int i)
        {
            return (*bottom_shape_)[channel_axis_ + i];
        }
        // reverse_dimensions should return true iff we are implementing deconv, so
        // that conv helpers know which dimensions are which.
        virtual bool reverse_dimensions() = 0;
        // Compute height_out_ and width_out_ from other parameters.
        virtual void compute_output_shape() = 0;

        /// @brief The spatial dimensions of a filter kernel.
        /// 卷积核尺寸[k_h,k_w]
        Blob<int> kernel_shape_;
        /// @brief The spatial dimensions of the stride.
        Blob<int> stride_;
        /// @brief The spatial dimensions of the padding.
        Blob<int> pad_;
        /// @brief The spatial dimensions of the dilation.
        Blob<int> dilation_;
        /// @brief The spatial dimensions of the convolution input.
        /// 输入特征图的空间尺寸 [c_in,f_h,f_w]
        Blob<int> conv_input_shape_;
        /// @brief The spatial dimensions of the col_buffer.
        vector<int> col_buffer_shape_; //一个输出通道对应的所有卷积核的所有卷积区域转化成一列向量的形状
        /// @brief The spatial dimensions of the output.
        vector<int> output_shape_;
        const vector<int> *bottom_shape_;

        int num_spatial_axes_;
        int bottom_dim_; // 表明每个batch的偏移量,总共有num_个batch
        int top_dim_;    // 表明每个batch的偏移量,总共有num_个batch

        int channel_axis_; //channel的axis
        int num_;          //bottom[0]->count(0, channel_axis_)  ----->  batch size
        int channels_;
        int group_;
        int out_spatial_dim_;
        int weight_offset_;
        int num_output_;
        bool bias_term_;
        bool is_1x1_;
        bool force_nd_im2col_;

    private:
        // wrap im2col/col2im so we don't have to remember the (long) argument lists
        inline void conv_im2col_cpu(const Dtype *data, Dtype *col_buff)
        {
            if (!force_nd_im2col_ && num_spatial_axes_ == 2)
            {
                // 对数据转换成im2col
                im2col_cpu(data, conv_in_channels_,
                           conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                           kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                           pad_.cpu_data()[0], pad_.cpu_data()[1],
                           stride_.cpu_data()[0], stride_.cpu_data()[1],
                           dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
            }
            else
            {
                im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
                              col_buffer_shape_.data(), kernel_shape_.cpu_data(),
                              pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
            }
        }

#ifndef CPU_ONLY
        inline void conv_im2col_gpu(const Dtype *data, Dtype *col_buff)
        {
            if (!force_nd_im2col_ && num_spatial_axes_ == 2)
            {
                // 对数据转换成im2col
                im2col_gpu(data, conv_in_channels_,
                           conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
                           kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
                           pad_.cpu_data()[0], pad_.cpu_data()[1],
                           stride_.cpu_data()[0], stride_.cpu_data()[1],
                           dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
            }
            else
            {
                im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
                              conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
                              kernel_shape_.gpu_data(), pad_.gpu_data(),
                              stride_.gpu_data(), dilation_.gpu_data(), col_buff);
            }
        }
#endif //CPU_ONLY

        int num_kernels_im2col_;
        int conv_out_channels_;
        int conv_in_channels_;
        int conv_out_spatial_dim_;
        int kernel_dim_;
        int col_offset_;
        int output_offset_;

        Blob<Dtype> col_buffer_; //im2col的分配空间
        Blob<Dtype> bias_multiplier_;
    };

} //namespace caffe

#endif //CAFFE_BASE_CONVOLUTION_LAYER_HPP_