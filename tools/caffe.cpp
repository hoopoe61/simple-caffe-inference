#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;

#ifndef USE_OPENCV
int main(int argc, char **argv)
{
    string model = "/home/dengshunge/Desktop/mycaffe/model/resnet18_deploy.prototxt";
    string weights = "/home/dengshunge/Desktop/mycaffe/model/solver_iter.caffemodel";

    std::cout << "Use CPU" << std::endl;
    Caffe::set_mode(Caffe::CPU);

    Net<float> caffe_net(model, caffe::TEST, 0, nullptr);
    caffe_net.CopyTrainedLayersFrom(weights);
    std::cout << "Finish loading" << std::endl;

    std::cout << "input name : " << caffe_net.layer_names()[0] << std::endl;
    std::cout << "blob name : " << caffe_net.blob_names()[0] << std::endl;
    std::cout << caffe_net.num_outputs() << std::endl;
    // const vector<Blob<float> *> &result = caffe_net.Forward();
    // caffe_net.Forward();
}
#else
using namespace caffe; // NOLINT(build/namespaces)
using std::string;

class Classifier
{
public:
    Classifier(const string &model_file,
               const string &trained_file);

    void Classify(const cv::Mat &img);

private:
    std::vector<float> Predict(const cv::Mat &img);

    void WrapInputLayer(std::vector<cv::Mat> *input_channels);

    void Preprocess(const cv::Mat &img,
                    std::vector<cv::Mat> *input_channels);

private:
    shared_ptr<Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
};

Classifier::Classifier(const string &model_file,
                       const string &trained_file)
{
    Caffe::set_mode(Caffe::CPU);

    /* Load the network. */
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float> *input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

/* Return the top N predictions. */
void Classifier::Classify(const cv::Mat &img)
{
    std::vector<float> output = Predict(img);

    for (auto c : output)
        std::cout << c << " ";
    std::cout << endl;
}

std::vector<float> Classifier::Predict(const cv::Mat &img)
{
    Blob<float> *input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = net_->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat> *input_channels)
{
    Blob<float> *input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat &img,
                            std::vector<cv::Mat> *input_channels)
{
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    // cv::Mat sample_normalized;
    // cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
    // cv::split(sample_normalized, *input_channels); //这里讲BGR通道分离,然后直接写入到blob中
    cv::split(sample_float, *input_channels); //这里讲BGR通道分离,然后直接写入到blob中

    CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
}

int main()
{
    string model = "/mycaffe/model/resnet18_deploy.prototxt";
    string weights = "/mycaffe/model/solver_iter.caffemodel";

    Classifier classifier(model, weights);
    cv::Mat img = cv::imread("/mycaffe/model/20200804-09-12-40-120_0.jpg", -1);
    CHECK(!img.empty()) << "Unable to decode image ";
    classifier.Classify(img);
}

#endif