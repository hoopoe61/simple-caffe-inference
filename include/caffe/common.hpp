#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/thread.hpp>

namespace caffe
{

    class Caffe
    {
    public:
        ~Caffe();
        Caffe(const Caffe &) = delete;            //阻止拷贝
        Caffe &operator=(const Caffe &) = delete; //阻止赋值

        static Caffe &Get();

        enum Brew
        {
            CPU,
            GPU
        };

        inline static void set_mode(Brew mode) { Get().mode_ = mode; }

    protected:
        Brew mode_;

    private:
        Caffe();
    };
} // namespace caffe

#endif //CAFFE_COMMON_HPP_
