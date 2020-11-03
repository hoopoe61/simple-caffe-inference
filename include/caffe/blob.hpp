#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;

namespace caffe
{
  template <typename Dtype>
  class Blob
  {
  public:
    Blob()
        : data_(), count_(0), capacity_(0){};
    explicit Blob(const vector<int> &shape); //创建形状为shape的blob
    // Blob(const Blob &) = delete;
    // Blob &operator=(const Blob &) = delete;
    // ~Blob() = default;

    /**
     * @brief 改变blob的维度,必要时分配新内存
     * 
     * 这个函数可以在初始化分配内存是调用(即在Blob()中调用),或者用于调整top blob的维度(在
     * Layer::Reshape 或者 Layer::Forward)
    **/
    void Reshape(const vector<int> &shape);
    void Reshape(const BlobShape &shape);
    void ReshapeLike(const Blob &other);
    // 用于输出形状信息
    inline string shape_string() const
    {
      ostringstream stream;
      for (int i = 0; i < shape_.size(); ++i)
      {
        stream << shape_[i] << " ";
      }
      stream << "(" << count_ << ")";
      return stream.str();
    }
    inline const vector<int> &shape() const { return shape_; }
    /**
     * @brief Returns the dimension of the index-th axis (or the negative index-th
     *        axis from the end, if index is negative).
     *
     * @param index the axis index, which may be negative as it will be
     *        "canonicalized" using CanonicalAxisIndex.
     *        Dies on out of range index.
     */
    inline int shape(int index) const
    {
      return shape_[CanonicalAxisIndex(index)];
    }
    // 偏移量计算
    inline int offset(const int n, const int c = 0, const int h = 0,
                      const int w = 0) const
    {
      CHECK_GE(n, 0);
      CHECK_LE(n, shape(0));
      CHECK_GE(shape(1), 0);
      CHECK_LE(c, shape(1));
      CHECK_GE(shape(2), 0);
      CHECK_LE(h, shape(2));
      CHECK_GE(shape(3), 0);
      CHECK_LE(w, shape(3));
      return ((n * shape(1) + c) * shape(2) + h) * shape(3) + w;
    }

    inline int offset(const vector<int> &indices) const
    {
      CHECK_LE(indices.size(), num_axes());
      int offset = 0;
      for (int i = 0; i < num_axes(); ++i)
      {
        offset *= shape(i);
        if (indices.size() > i)
        {
          CHECK_GE(indices[i], 0);
          CHECK_LT(indices[i], shape(i));
          offset += indices[i];
        }
      }
      return offset;
    }

    inline int num_axes() const { return shape_.size(); }
    inline int count() const { return count_; }

    /**计算切片的容量
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *        计算切片的容量
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */
    inline int count(int start_axis, int end_axis) const
    {
      CHECK_LE(start_axis, end_axis);
      CHECK_GE(start_axis, 0);
      CHECK_GE(end_axis, 0);
      CHECK_LE(start_axis, num_axes());
      CHECK_LE(end_axis, num_axes());
      int count = 1;
      for (int i = start_axis; i < end_axis; ++i)
      {
        count *= shape(i);
      }
      return count;
    }
    /**计算从特定轴开始的切片容量
     * @brief Compute the volume of a slice spanning from a particular first
     *        axis to the final axis.
     *
     * @param start_axis The first axis to include in the slice.
     */
    inline int count(int start_axis) const
    {
      return count(start_axis, num_axes());
    }

    /**返回canonical版本的index，类似于numpy中的-1
     * @brief Returns the 'canonical' version of a (usually) user-specified axis,
     *        allowing for negative indexing (e.g., -1 for the last axis).
     *
     * @param axis_index the axis index.
     *        If 0 <= index < num_axes(), return index.
     *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
     *        e.g., the last axis index (num_axes() - 1) if index == -1,
     *        the second to last if index == -2, etc.
     *        Dies on out of range index.
     */
    inline int CanonicalAxisIndex(int axis_index) const
    {
      CHECK_GE(axis_index, -num_axes())
          << "axis " << axis_index << " out of range for " << num_axes()
          << "-D Blob with shape " << shape_string();
      CHECK_LT(axis_index, num_axes())
          << "axis " << axis_index << " out of range for " << num_axes()
          << "-D Blob with shape " << shape_string();
      if (axis_index < 0)
      {
        return axis_index + num_axes();
      }
      return axis_index;
    }

    inline const shared_ptr<SyncedMemory> &data() const
    {
      CHECK(data_);
      return data_;
    }

    const Dtype *cpu_data() const;
    void set_cpu_data(Dtype *data);
    Dtype *mutable_cpu_data();
    void FromProto(const BlobProto &proto, bool reshape = true); //从另外一个blob中读取数据

    /**
     * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
     *        data_ of Blob other -- useful in Layer%s which simply perform a copy
     *        in their Forward pass.
     *      通过设置data_的智能指针，指向另一个blob掌握的data_，以此来实现内容共享。
     * This deallocates the SyncedMemory holding this Blob's data_, as
     * shared_ptr calls its destructor when reset with the "=" operator.
     */
    void ShareData(const Blob &other);

    bool ShapeEquals(const BlobProto &other); //判断两个blob是否具有相同的维度

  private:
    shared_ptr<SyncedMemory> data_;       //数据
    shared_ptr<SyncedMemory> shape_data_; //blob的shape
    vector<int> shape_;
    int count_;    //大小为N*C*H*W
    int capacity_; //最大容量

    DISABLE_COPY_AND_ASSIGN(Blob);
  }; // class Blob

} // namespace caffe

#endif //CAFFE_BLOB_HPP_