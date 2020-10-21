#include "include/caffe/caffe.hpp"

// using caffe::Blob;
using caffe::Caffe;
// using caffe::Layer;
// using caffe::Net;
// using caffe::shared_ptr;

using namespace std;

int main(int argc, char **argv)
{
    string model = "";
    string weights = "";

    cout << "Use CPU" << endl;
    Caffe::set_mode(Caffe::CPU);

    // Net<float> caffe_net(model,caffe::Test);
    // caffe_net.CopyTrainedLayersFrom(weights);

    // const vector<Blob<float> *> &result = caffe_net.Forward();



    return 0;
}
