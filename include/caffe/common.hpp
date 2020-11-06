#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <iostream>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream> // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility> // pair
#include <vector>

// gflags 2.1 issue: namespace google was changed to gflags without warning.
// Luckily we will be able to use GFLAGS_GFLAGS_H_ to detect if it is version
// 2.1. If yes, we will add a temporary solution to redirect the namespace.
// TODO(Yangqing): Once gflags solves the problem in a more elegant way, let's
// remove the following hack.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
// 禁止拷贝复制与移动复制
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:                                   \
    classname(const classname &) = delete; \
    classname &operator=(const classname &) = delete

// Instantiate a class with float and double specifications.
// 实例化类
#define INSTANTIATE_CLASS(classname)     \
    char gInstantiationGuard##classname; \
    template class classname<float>;     \
    template class classname<double>

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

namespace caffe
{
    // We will use the boost shared_ptr instead of the new C++11 one mainly
    // because cuda does not work (at least now) well with C++11 features.
    using boost::shared_ptr;

    // Common functions and classes from std that caffe often uses.
    using std::cout;
    using std::endl;
    using std::fstream;
    using std::ios;
    using std::isinf;
    using std::isnan;
    using std::iterator;
    using std::make_pair;
    using std::map;
    using std::ostringstream;
    using std::pair;
    using std::set;
    using std::string;
    using std::stringstream;
    using std::vector;

    class Caffe
    {
    public:
        ~Caffe();
        static Caffe &Get();

        enum Brew
        {
            CPU,
            GPU
        };

        // Returns the mode: running on CPU or GPU.
        inline static Brew mode() { return Get().mode_; }
        // The setters for the variables
        // Sets the mode. It is recommended that you don't change the mode halfway
        // into the program since that may cause allocation of pinned memory being
        // freed in a non-pinned way, which may cause problems - I haven't verified
        // it personally but better to note it here in the header file.
        inline static void set_mode(Brew mode) { Get().mode_ = mode; }
        inline static void set_solver_rank(int val) { Get().solver_rank_ = val; } //用于控制是否输出LOG信息
        inline static bool root_solver() { return Get().solver_rank_ == 0; }      //用于控制是否输出LOG信息

    protected:
        Brew mode_;
        int solver_rank_;

    private:
        Caffe();
        DISABLE_COPY_AND_ASSIGN(Caffe);
    };

} // namespace caffe

#endif // CAFFE_COMMON_HPP_