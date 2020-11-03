#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX; // Max size of 2 GB minus 1 byte.

namespace caffe
{
    using google::protobuf::Message;
    using google::protobuf::io::CodedInputStream;
    using google::protobuf::io::FileInputStream;
    using google::protobuf::io::ZeroCopyInputStream;

    bool ReadProtoFromTextFile(const char *filename, Message *proto)
    {
        int fd = open(filename, O_RDONLY);
        CHECK_NE(fd, -1) << "File not found: " << filename;
        FileInputStream *input = new FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }

    bool ReadProtoFromBinaryFile(const char *filename, Message *proto)
    {
        int fd = open(filename, O_RDONLY);
        CHECK_NE(fd, -1) << "File not found: " << filename;
        ZeroCopyInputStream *raw_input = new FileInputStream(fd);
        CodedInputStream *coded_input = new CodedInputStream(raw_input);
        coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

        bool success = proto->ParseFromCodedStream(coded_input);

        delete coded_input;
        delete raw_input;
        close(fd);
        return success;
    }

} // namespace caffe