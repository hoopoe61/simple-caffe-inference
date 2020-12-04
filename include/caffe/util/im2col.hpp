#ifndef _CAFFE_UTIL_IM2COL_HPP_
#define _CAFFE_UTIL_IM2COL_HPP_

/*
* 结合网站 https://blog.csdn.net/jiongnima/article/details/69736844 更容易理解
*/

namespace caffe
{
    template <typename Dtype>
    void im2col_nd_cpu(const Dtype *data_im, const int num_spatial_axes,
                       const int *im_shape, const int *col_shape,
                       const int *kernel_shape, const int *pad, const int *stride,
                       const int *dilation, Dtype *data_col);

    // 将data_im的数据进行im2col转换,放在data_col中
    template <typename Dtype>
    void im2col_cpu(const Dtype *data_im, const int channels,
                    const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h, const int pad_w, const int stride_h,
                    const int stride_w, const int dilation_h, const int dilation_w,
                    Dtype *data_col);

    template <typename Dtype>
    void im2col_nd_gpu(const Dtype *data_im, const int num_spatial_axes,
                       const int col_size, const int *im_shape, const int *col_shape,
                       const int *kernel_shape, const int *pad, const int *stride,
                       const int *dilation, Dtype *data_col);

    template <typename Dtype>
    void im2col_gpu(const Dtype *data_im, const int channels,
                    const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h, const int pad_w, const int stride_h,
                    const int stride_w, const int dilation_h, const int dilation_w,
                    Dtype *data_col);
} // namespace caffe

#endif //_CAFFE_UTIL_IM2COL_HPP_