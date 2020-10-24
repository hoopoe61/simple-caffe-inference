#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;
using caffe::shared_ptr;
using caffe::Solver;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

int main(int argc, char **argv)
{
    string model = "/home/dengshunge/Desktop/PyCharm_python3/NSPP/resnet18/使用sigmoid_改变标签编码方式/log/caffe/resnet18_deploy.prototxt";
    string weights = "/home/dengshunge/Desktop/PyCharm_python3/NSPP/resnet18/使用sigmoid_改变标签编码方式/log/caffe/solver_iter.caffemodel";

    std::cout << "Use CPU" << std::endl;
    Caffe::set_mode(Caffe::CPU);

    Net<float> caffe_net(model, caffe::TEST, 0, nullptr);
    caffe_net.CopyTrainedLayersFrom(weights);
    std::cout << "Finish loading" << std::endl;

    // const vector<Blob<float> *> &result = caffe_net.Forward();
    caffe_net.Forward();
}
