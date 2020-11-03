#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"

namespace caffe
{
    using ::boost::filesystem::path;
    using ::google::protobuf::Message;

    // 从text中读取Proto
    bool ReadProtoFromTextFile(const char *filename, Message *proto);

    inline bool ReadProtoFromTextFile(const string &filename, Message *proto)
    {
        return ReadProtoFromTextFile(filename.c_str(), proto);
    }

    // 从二进制文件中读取Proto
    bool ReadProtoFromBinaryFile(const char *filename, Message *proto);

    inline bool ReadProtoFromBinaryFile(const string &filename, Message *proto)
    {
        return ReadProtoFromBinaryFile(filename.c_str(), proto);
    }

} // namespace caffe

#endif //CAFFE_UTIL_IO_H_