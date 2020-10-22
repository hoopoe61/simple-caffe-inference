#ifndef CAFFE_NET_HPP_
#define CAFFE_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/common.hpp"

namespace caffe
{
    template <typename Dtype>
    class Net
    {
    public:
        // Net(const std::string &param_file, Phase phase, const int level = 0, const vector<string> *stages = NULL);
        virtual ~Net() {}

        /// @brief Initialize a network with a NetParameter.
        // void Init(const NetParameter &param);

        ///@brief Run Forward and return the result.
        // const vector<Blob<Dtype> *> &Forward(Dtype *loss = NULL);

    protected:
        /// @brief The network name
        std::string name_;
        /// @brief The phase: TRAIN or TEST
        // Phase phase_;
    };
} // namespace caffe

#endif