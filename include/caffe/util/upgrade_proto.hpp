#ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
#define CAFFE_UTIL_UPGRADE_PROTO_H_

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace caffe
{
    // Check for deprecations and upgrade the NetParameter as needed.
    bool UpgradeNetAsNeeded(const string &param_file, NetParameter *param);

    // Read parameters from a file into a NetParameter proto message.
    void ReadNetParamsFromTextFileOrDie(const string &param_file,
                                        NetParameter *param);
    void ReadNetParamsFromBinaryFileOrDie(const string &param_file,
                                          NetParameter *param);

    // Return true iff the Net contains input fields.
    bool NetNeedsInputUpgrade(const NetParameter &net_param);
    // Perform all necessary transformations to upgrade input fields into layers.
    void UpgradeNetInput(NetParameter *net_param);
    // Return true iff the Net contains batch norm layers with manual local LRs.
    bool NetNeedsBatchNormUpgrade(const NetParameter &net_param);
    // Perform all necessary transformations to upgrade batch norm layers.
    void UpgradeNetBatchNorm(NetParameter *net_param);

} // namespace caffe

#endif //CAFFE_UTIL_UPGRADE_PROTO_H_