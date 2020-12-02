#include "caffe/blob.hpp"

namespace caffe
{
  //创建shape形状的blob
  template <typename Dtype>
  Blob<Dtype>::Blob(const vector<int> &shape)
      // 在调用Blob::Reshape()函数之前,capacity_必须初始化
      : capacity_(0)
  {
    Reshape(shape);
  }

  template <typename Dtype>
  void Blob<Dtype>::Reshape(const vector<int> &shape)
  {
    CHECK_LE(shape.size(), kMaxBlobAxes);
    count_ = 1;
    shape_.resize(shape.size());
    if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int))
    {
      //如果shape_data_的数据为空,或者尺寸过小,则重新分配
      //TODO 这里有点问题,如果尺寸过大,就不理会了?
      shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
    }
    int *shape_data = static_cast<int *>(shape_data_->mutable_cpu_data());
    for (int i = 0; i < shape.size(); ++i)
    {
      CHECK_GE(shape[i], 0); //需要每个shape都大于0
      if (count_ != 0)
      {
        // 因为count_表示元素个数(N*B*H*W),防止越界
        CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
      }
      count_ *= shape[i];
      shape_[i] = shape[i];
      shape_data[i] = shape[i]; //放入到SyncedMemory的内存中
    }
    if (count_ > capacity_)
    {
      capacity_ = count_;
      data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype))); //分配新的空间
    }
  }

  template <typename Dtype>
  void Blob<Dtype>::Reshape(const BlobShape &shape)
  {
    CHECK_LE(shape.dim_size(), kMaxBlobAxes);
    vector<int> shape_vec(shape.dim_size());
    for (int i = 0; i < shape.dim_size(); ++i)
    {
      shape_vec[i] = shape.dim(i);
    }
    Reshape(shape_vec);
  }

  template <typename Dtype>
  void Blob<Dtype>::ReshapeLike(const Blob &other)
  {
    Reshape(other.shape());
  }

  template <typename Dtype>
  const Dtype *Blob<Dtype>::cpu_data() const
  {
    CHECK(data_);
    return (const Dtype *)data_->cpu_data();
  }

  template <typename Dtype>
  void Blob<Dtype>::set_cpu_data(Dtype *data)
  {
    CHECK(data); //防止空指针
    // 保证尺寸相同
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size)
    {
      data_.reset(new SyncedMemory(size));
    }
    data_->set_cpu_data(data);
  }

  template <typename Dtype>
  Dtype *Blob<Dtype>::mutable_cpu_data()
  {
    CHECK(data_);
    return static_cast<Dtype *>(data_->mutable_cpu_data());
  }

  template <typename Dtype>
  const int *Blob<Dtype>::gpu_shape() const
  {
    //同步至gpu的数据
    CHECK(shape_data_);
    return (const int *)shape_data_->gpu_data();
  }

  template <typename Dtype>
  const Dtype *Blob<Dtype>::gpu_data() const
  {
    CHECK(data_);
    return (const Dtype *)data_->gpu_data();
  }

  template <typename Dtype>
  void Blob<Dtype>::set_gpu_data(Dtype *data)
  {
    CHECK(data);
    // Make sure CPU and GPU sizes remain equal
    size_t size = count_ * sizeof(Dtype);
    if (data_->size() != size)
    {
      data_.reset(new SyncedMemory(size));
    }
    data_->set_gpu_data(data);
  }

  template <typename Dtype>
  Dtype *Blob<Dtype>::mutable_gpu_data()
  {
    CHECK(data_);
    return static_cast<Dtype *>(data_->mutable_gpu_data());
  }

  template <typename Dtype>
  void Blob<Dtype>::ShareData(const Blob &other)
  {
    CHECK_EQ(count_, other.count());
    data_ = other.data();
  }

  template <typename Dtype>
  bool Blob<Dtype>::ShapeEquals(const BlobProto &other)
  {
    //判断两个blob的shape是否相同
    vector<int> other_shape(other.shape().dim_size());
    for (int i = 0; i < other.shape().dim_size(); ++i)
    {
      other_shape[i] = other.shape().dim(i);
    }
    return shape_ == other_shape;
  }

  template <typename Dtype>
  void Blob<Dtype>::FromProto(const BlobProto &proto, bool reshape)
  {
    if (reshape)
    {
      // 注:这里不支持旧式的blob,即other.has_num() || other.has_channels() ||other.has_height() || other.has_width()
      vector<int> shape;
      shape.resize(proto.shape().dim_size());
      for (int i = 0; i < proto.shape().dim_size(); ++i)
      {
        shape[i] = proto.shape().dim(i);
      }
      Reshape(shape);
    }
    else
    {
      CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
    }
    // copy data
    Dtype *data_vec = mutable_cpu_data();
    if (proto.double_data_size() > 0)
    {
      //数据类别是double
      CHECK_EQ(count_, proto.double_data_size()); //检查特征图容量与数据容量是否相等
      for (int i = 0; i < count_; ++i)
      {
        data_vec[i] = proto.double_data(i);
      }
    }
    else
    {
      CHECK_EQ(count_, proto.data_size());
      for (int i = 0; i < count_; ++i)
      {
        data_vec[i] = proto.data(i);
      }
    }
  }

  INSTANTIATE_CLASS(Blob);
  template class Blob<int>;
  template class Blob<unsigned int>;

} //namespace caffe