#include "caffe/caffe.hpp"
#include <string>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

using caffe::Blob;
using caffe::Caffe;
using caffe::Layer;
using caffe::Net;
using caffe::shared_ptr;
using caffe::string;
using caffe::vector;
using std::cout;
using std::endl;
using std::ostringstream;

#ifndef USE_OPENCV
int main(int argc, char **argv)
{
    string model = "/home/dengshunge/Desktop/mycaffe/model/resnet18_deploy.prototxt";
    string weights = "/home/dengshunge/Desktop/mycaffe/model/solver_iter.caffemodel";

    //设置工作模式:CPU or GPU
    std::cout << "Use CPU" << std::endl;
    Caffe::set_mode(Caffe::CPU);
    Caffe::set_solver_rank(1); //不进行日志输出

    Net<float> caffe_net(model, caffe::TEST, 0, nullptr);
    caffe_net.CopyTrainedLayersFrom(weights);
    std::cout << "Finish loading" << std::endl;

    Blob<float> *input_layer = caffe_net.input_blobs()[0];
    input_layer->Reshape(vector<int>{1, 3, 224, 112});
    caffe_net.Reshape();

    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < 3 * 224 * 112; ++i)
        input_data[i] = i * 1.0;

    // const float *input_data_ = input_layer->cpu_data();
    // for (int channel = 0; channel < 3; ++channel)
    // {
    //     for (int row = 0; row < 10; ++row)
    //     {
    //         for (int col = 0; col < 10; ++col)
    //         {
    //             cout << input_data_[channel * 224 * 112 + row * 112 + col] << " ";
    //         }
    //         cout << endl;
    //     }
    //     cout << "---------------" << endl;
    // }
    // for (int i = 0; i < 20; ++i)
    //     cout << input_data_[i] << " ";
    // cout << endl;

    caffe_net.Forward();
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
    num_channels_ = input_layer->shape(1);
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = cv::Size(input_layer->shape(3), input_layer->shape(2));
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
    input_layer->Reshape(vector<int>{1, num_channels_, input_geometry_.height, input_geometry_.width});
    /* Forward dimension change to all layers. */
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    Preprocess(img, &input_channels);

    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = net_->output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->shape(1);
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

    int width = input_layer->shape(3);
    int height = input_layer->shape(2);
    float *input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->shape(1); ++i)
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
    cout << "true = [0.00210892 1.09131e-07]" << endl;
}

#endif