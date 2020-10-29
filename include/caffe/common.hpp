#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <iostream>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

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
    //禁止拷贝构造与移动构造
    Caffe(const Caffe &) = delete;
    Caffe &operator=(const Caffe &) = delete;

    ~Caffe();
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