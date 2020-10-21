#include "caffe/common.hpp"

namespace caffe
{
    // Make sure each thread can have different values.
    static boost::thread_specific_ptr<Caffe> thread_instance_;

    Caffe &Caffe::Get()
    {
        if (!thread_instance_.get())
        {
            thread_instance_.reset(new Caffe());
        }
        return *(thread_instance_.get());
    }

    Caffe::Caffe() : mode_(Caffe::CPU)
    {
    }

    Caffe::~Caffe() {}
} // namespace caffe
