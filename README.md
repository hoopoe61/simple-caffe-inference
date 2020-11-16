# simple-Caffe-Inference

#### 介绍
此项目是基于原版caffe进行精简和重构，值具有CPU推理功能，用于了解推理框架的原理。
请参考我的博客：[xxx](https://www.cnblogs.com/dengshunge/)
此简化的SimpleCaffe具有以下特点：
1. 只有CPU推理功能，无需GPU；
2. 只有前向计算能力，无后向求导功能；
3. 接口保持与原版的Caffe一致；
4. 精简了大部分代码，并进行了详尽注释。

#### 安装说明
由于本项目是基于原版caffe的，所以需要在物理机上安装上caffe所需要的库。
当安装完相应的库后，可以进行如下方式进行编译：

```
git clone https://gitee.com/dengshunge/simple-caffe-inference.git
cd simple-caffe-inference
// 修改需求修改根目录下的CMakeLists.txt文件，主要就是修改其中的caffe_option选项，
mkdir build
cd build
cmake ..
make -j4
```

当编译完成后，使用下面代码来运行

```
//位于simple-caffe-inference/build目录下
./tool/caffe
```

#### 本机环境
本物理机是Ubuntu16.04，可以正常运行。

