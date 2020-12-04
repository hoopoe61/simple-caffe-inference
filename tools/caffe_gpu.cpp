#include "caffe/caffe.hpp"
#include <string>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif // USE_OPENCV

// 用于计时
#include <boost/date_time/posix_time/posix_time.hpp>

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
    string model = "../model/deploy.prototxt";
    string weights = "../model/squeezenet_v1.1.caffemodel";

    //设置工作模式:CPU or GPU
    std::cout << "Use GPU" << std::endl;
    // ::google::InitGoogleLogging("caffe"); //初始化日志文件,不调用会给出警告,但不会报错
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_rank(1); //不进行日志输出

    Net<float> caffe_net(model, caffe::TEST, 0, nullptr);
    caffe_net.CopyTrainedLayersFrom(weights);
    std::cout << "Finish loading" << std::endl;

    // Blob<float> *input_layer = caffe_net.input_blobs()[0];
    // input_layer->Reshape(vector<int>{1, 3, 227, 227});
    // caffe_net.Reshape();

    boost::posix_time::ptime start_time_ = boost::posix_time::microsec_clock::local_time(); //开始计时

    for (int i = 0; i < 1000; ++i)
        caffe_net.Forward();

    boost::posix_time::ptime end_time_ = boost::posix_time::microsec_clock::local_time(); //结束计时
    std::cout << "average time : " << (end_time_ - start_time_).total_milliseconds() / 1000. << " ms" << std::endl;
}

#else
using namespace caffe; // NOLINT(build/namespaces)
using std::string;

void Preprocess(cv::Mat &img, const int width, const int height,
                const vector<float> &mean, const vector<float> &scale,
                float *input_data)
{
    cv::resize(img, img, cv::Size(width, height));
    img.convertTo(img, CV_32FC3);

    // 将通道split的结果直接放在input_data的地址上,减少一次拷贝
    vector<cv::Mat> bgrChannels;
    for (int i = 0; i < 3; ++i)
    {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        bgrChannels.push_back(channel);
        input_data += width * height;
    }
    cv::split(img, bgrChannels);

    // 归一化处理
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            bgrChannels[0].at<float>(i, j) = (bgrChannels[0].at<float>(i, j) - mean[0]) * scale[0];
            bgrChannels[1].at<float>(i, j) = (bgrChannels[1].at<float>(i, j) - mean[1]) * scale[1];
            bgrChannels[2].at<float>(i, j) = (bgrChannels[2].at<float>(i, j) - mean[2]) * scale[2];
        }
    }
}

int main()
{
    string model = "../model/deploy.prototxt";
    string weights = "../model/squeezenet_v1.1.caffemodel";

    ::google::InitGoogleLogging("caffe"); //初始化日志文件,不调用会给出警告,但不会报错
    Caffe::set_mode(Caffe::GPU);
    Caffe::set_solver_rank(1); //不进行日志输出
    Net<float> caffe_net(model, caffe::TEST, 0, nullptr);
    caffe_net.CopyTrainedLayersFrom(weights);

    // 读入图片
    cv::Mat img = cv::imread("../model/dog.jpg");
    CHECK(!img.empty()) << "Unable to decode image ";

    // 图片预处理,并加载图片进入blob
    Blob<float> *input_layer = caffe_net.input_blobs()[0];
    float *input_data = input_layer->mutable_cpu_data();
    const int height = 227;
    const int width = 227;
    vector<float> mean{104., 117., 123.};
    vector<float> scale{1.0, 1.0, 1.0};
    Preprocess(img, width, height, mean, scale, input_data);

    boost::posix_time::ptime start_time_ = boost::posix_time::microsec_clock::local_time(); //开始计时

    //前向运算
    for (int i = 0; i < 500; ++i)
        caffe_net.Forward();

    boost::posix_time::ptime end_time_ = boost::posix_time::microsec_clock::local_time(); //结束计时

    /* Copy the output layer to a std::vector */
    Blob<float> *output_layer = caffe_net.output_blobs()[0];
    const float *begin = output_layer->cpu_data();
    const float *end = begin + output_layer->shape(1);
    vector<float> results(begin, end);

    cout << "true index = 263 prob = 0.8277238" << endl;
    cout << "pred : " << results[263] << endl;
    std::cout << "average time : " << (end_time_ - start_time_).total_milliseconds() / 500. << " ms" << std::endl;
}

#endif