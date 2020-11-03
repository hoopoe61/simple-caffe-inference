#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe
{
    // // Get convolution layer according to engine.
    // template <typename Dtype>
    // shared_ptr<Layer<Dtype>> GetConvolutionLayer(
    //     const LayerParameter &param)
    // {
    //     ConvolutionParameter conv_param = param.convolution_param();
    //     ConvolutionParameter_Engine engine = conv_param.engine();
    //     if (engine == ConvolutionParameter_Engine_DEFAULT)
    //     {
    //         engine = ConvolutionParameter_Engine_CAFFE;
    //     }
    //     if (engine == ConvolutionParameter_Engine_CAFFE)
    //     {
    //         return shared_ptr<Layer<Dtype>>(new ConvolutionLayer<Dtype>(param));
    //     }
    //     else
    //     {
    //         LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    //         throw; // Avoids missing return warning
    //     }
    // }

    // REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);

} // namespace caffe