#ifndef CAFFE_LAYER_HPP_
#define CAFFE_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
    template <typename Dtype>
    class Layer
    {
    public:
        explicit Layer(const LayerParameter &param)
            : layer_param_(param), phase_(param.phase()) {}
        Layer(const Layer &) = delete;
        Layer &operator=(const Layer &) = delete;
        virtual ~Layer() {}

        /**
         * @brief Implements common layer setup functionality.
         *
         * @param bottom the preshaped input blobs
         * @param top
         *     the allocated but unshaped output blobs, to be shaped by Reshape
         *
         * Checks that the number of bottom and top blobs is correct.
         * Calls LayerSetUp to do special layer setup for individual layer types,
         * followed by Reshape to set up sizes of top blobs and internal buffers.
         * Sets up the loss weight multiplier blobs for any non-zero loss weights.
         * This method may not be overridden.
         * 检查bottom和top的blobs的数量是否正确
         * 调用LayerSetUp函数来创建特定的层
         * 调动Reshape函数来top blobs和中间buffers的尺寸
        **/
        void SetUp(const vector<Blob<Dtype> *> &bottom,
                   const vector<Blob<Dtype> *> &top)
        {
            CheckBlobCounts(bottom, top); //检查输入输出是否符合规定
            LayerSetUp(bottom, top);      //虚函数,只执行一次,对该层进行特定操作
            Reshape(bottom, top);         //虚函数,对输出的特征设置维度
        }

        /**
         * @brief Does layer-specific setup: your layer should implement this function
         *        as well as Reshape.
         *
         * @param bottom
         *     the preshaped input blobs, whose data fields store the input data for
         *     this layer
         * @param top
         *     the allocated but unshaped output blobs
         *
         * This method should do one-time layer specific setup. This includes reading
         * and processing relevent parameters from the <code>layer_param_</code>.
         * Setting up the shapes of top blobs and internal buffers should be done in
         * <code>Reshape</code>, which will be called before the forward pass to
         * adjust the top blob sizes.
         */
        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top) {}

        /**
         * @brief Adjust the shapes of top blobs and internal buffers to accommodate
         *        the shapes of the bottom blobs.
         * 调整top blobs和中间buffers的形状
         *
         * @param bottom the input blobs, with the requested input shapes
         * @param top the top blobs, which should be reshaped as needed
         *
         * This method should reshape top blobs as needed according to the shapes
         * of the bottom (input) blobs, as well as reshaping any internal buffers
         * and making any other necessary adjustments so that the layer can
         * accommodate the bottom blobs.
         */
        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top) = 0; //纯虚函数

        /**
         * @brief Given the bottom blobs, compute the top blobs and the loss.
         *          给定bottom blobs,计算top blobs和loss
         * @param bottom
         *     the input blobs, whose data fields store the input data for this layer
         * @param top
         *     the preshaped output blobs, whose data fields will store this layers'
         *     outputs
         * \return The total loss from the layer.
         *
         * The Forward wrapper calls the relevant device wrapper function
         * (Forward_cpu or Forward_gpu) to compute the top blob values given the
         * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
         * then computes and returns the loss.
         *
         * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
         */
        inline Dtype Forward(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        /**
         * @brief Returns the vector of learnable parameter blobs.
         */
        vector<shared_ptr<Blob<Dtype>>> &blobs()
        {
            return blobs_;
        }

        /**返回该层的参数
         * @brief Returns the layer parameter.
         */
        const LayerParameter &layer_param() const { return layer_param_; }

        /**
         * @brief Returns the layer type.
         */
        virtual inline const char *type() const { return ""; }
        /**
         * @brief Returns the exact number of bottom blobs required by the layer,
         *        or -1 if no exact number is required.
         *        返回该层所需的bottom blobs的确切数量
         * This method should be overridden to return a non-negative value if your
         * layer expects some exact number of bottom blobs.
         */
        virtual inline int ExactNumBottomBlobs() const { return -1; }

        /**
         * @brief Returns the minimum number of bottom blobs required by the layer,
         *        or -1 if no minimum number is required.
         *       返回该层所需的最少bottom blobs的数量
         * This method should be overridden to return a non-negative value if your
         * layer expects some minimum number of bottom blobs.
         */
        virtual inline int MinBottomBlobs() const { return -1; }

        /**
         * @brief Returns the maximum number of bottom blobs required by the layer,
         *        or -1 if no maximum number is required.
         *        返回该层所需的最多bottom blobs的数量
         * This method should be overridden to return a non-negative value if your
         * layer expects some maximum number of bottom blobs.
         */
        virtual inline int MaxBottomBlobs() const { return -1; }

        /**
         * @brief Returns the exact number of top blobs required by the layer,
         *        or -1 if no exact number is required.
         *        返回该层所需的确切top blobs的数量
         * This method should be overridden to return a non-negative value if your
         * layer expects some exact number of top blobs.
         */
        virtual inline int ExactNumTopBlobs() const { return -1; }

        /**
         * @brief Returns the minimum number of top blobs required by the layer,
         *        or -1 if no minimum number is required.
         *          返回该层所需的最少top blobs的数量
         * This method should be overridden to return a non-negative value if your
         * layer expects some minimum number of top blobs.
         */
        virtual inline int MinTopBlobs() const { return -1; }

        /**
         * @brief Returns the maximum number of top blobs required by the layer,
         *        or -1 if no maximum number is required.
         *          返回该层所需的嘴都top blobs的数量
         * This method should be overridden to return a non-negative value if your
         * layer expects some maximum number of top blobs.
         */
        virtual inline int MaxTopBlobs() const { return -1; }

        /**
         * @brief Returns true if the layer requires an equal number of bottom and
         *        top blobs.
         *        返回bottom blob的数量是否需要与top blob的数量一致
         * This method should be overridden to return true if your layer expects an
         * equal number of bottom and top blobs.
         */
        virtual inline bool EqualNumBottomTopBlobs() const { return false; }

        /**
         * @brief Return whether "anonymous" top blobs are created automatically
         *        by the layer.
         *          当为真时,会自动生成足够的blobs,以此来满足ExactNumTopBlobs()或者MinTopBlobs()对top blob的需求
         * If this method returns true, Net::Init will create enough "anonymous" top
         * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
         * MinTopBlobs().
         */
        virtual inline bool AutoTopBlobs() const { return false; }

    protected:
        //层参数
        LayerParameter layer_param_;
        //层状态 test
        Phase phase_;
        /** The vector that stores the learnable parameters as a set of blobs. */
        vector<shared_ptr<Blob<Dtype>>> blobs_; //存储可学习的参数

        /** @brief Using the CPU device, compute the layer output. */
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top) = 0;

        /**
         * @brief Using the GPU device, compute the layer output.
         *        Fall back to Forward_cpu() if unavailable.
         * 保留接口
         */
        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top)
        {
            // LOG(WARNING) << "Using CPU code as backup.";
            return Forward_cpu(bottom, top);
        }

        // //保留后传接口
        // virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
        //                           const vector<bool> &propagate_down,
        //                           const vector<Blob<Dtype> *> &bottom) = 0;
        // virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
        //                           const vector<bool> &propagate_down,
        //                           const vector<Blob<Dtype> *> &bottom)
        // {
        //     // LOG(WARNING) << "Using CPU code as backup.";
        //     Backward_cpu(top, propagate_down, bottom);
        // }

        /**
         * Called by the parent Layer's SetUp to check that the number of bottom
         * and top Blobs provided as input match the expected numbers specified by
         * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
         * 检查top blob和bottom blob的数量是否正确
         */
        virtual void CheckBlobCounts(const vector<Blob<Dtype> *> &bottom,
                                     const vector<Blob<Dtype> *> &top)
        {
            if (ExactNumBottomBlobs() >= 0)
            {
                CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
                    << type() << " Layer takes " << ExactNumBottomBlobs()
                    << " bottom blob(s) as input.";
            }
            if (MinBottomBlobs() >= 0)
            {
                CHECK_LE(MinBottomBlobs(), bottom.size())
                    << type() << " Layer takes at least " << MinBottomBlobs()
                    << " bottom blob(s) as input.";
            }
            if (MaxBottomBlobs() >= 0)
            {
                CHECK_GE(MaxBottomBlobs(), bottom.size())
                    << type() << " Layer takes at most " << MaxBottomBlobs()
                    << " bottom blob(s) as input.";
            }
            if (ExactNumTopBlobs() >= 0)
            {
                CHECK_EQ(ExactNumTopBlobs(), top.size())
                    << type() << " Layer produces " << ExactNumTopBlobs()
                    << " top blob(s) as output.";
            }
            if (MinTopBlobs() >= 0)
            {
                CHECK_LE(MinTopBlobs(), top.size())
                    << type() << " Layer produces at least " << MinTopBlobs()
                    << " top blob(s) as output.";
            }
            if (MaxTopBlobs() >= 0)
            {
                CHECK_GE(MaxTopBlobs(), top.size())
                    << type() << " Layer produces at most " << MaxTopBlobs()
                    << " top blob(s) as output.";
            }
            if (EqualNumBottomTopBlobs())
            {
                CHECK_EQ(bottom.size(), top.size())
                    << type() << " Layer produces one top blob as output for each "
                    << "bottom blob input.";
            }
        }

    }; //class Layer

    // 前传接口
    // Forward wrappers. You should implement the cpu and
    // gpu specific implementations instead, and should not change these
    // functions.
    template <typename Dtype>
    inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top)
    {
        Reshape(bottom, top);
        switch (Caffe::mode())
        {
        case Caffe::CPU:
        {
            Forward_cpu(bottom, top);
            break;
        }
        case Caffe::GPU:
        {
            Forward_gpu(bottom, top);
            break;
        }
        default:
            LOG(FATAL) << "Unknown caffe mode.";
        }
        return 0;
    }

} //namespace caffe

#endif //CAFFE_LAYER_HPP_