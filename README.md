# simple-Caffe-Inference

#### 介绍
此项目是基于原版caffe进行精简和重构，具有CPU推理功能和GPU推理功能，用于了解推理框架的原理。  
本项目包含两部分:CPU版本与GPU版本.  
如果是首次接触caffe的话,可以参考如下资料:
1. 阅读[基于CPU版本的项目](https://gitee.com/dengshunge/simple-caffe-inference/tree/CPU-v1.0/);
2. 请参考我的博客：[基于CPU版本的Caffe推理框架](https://www.cnblogs.com/dengshunge/p/13972872.html)  


此简化的SimpleCaffe具有以下特点：
1. 具有CPU与GPU推理功能；
2. 只有前向计算能力，无后向求导功能；
3. 接口保持与原版的Caffe一致；
4. 精简了大部分代码，并进行了详尽注释.

相比于CPU版本,此GPU版本的更新的内容包含了:
1. CPU数据与GPU数据的同步;
2. 每个自定义层的forward_gpu;
3. mathfunction的GPU版本;
4. cublas的宏定义与调用等.  

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
本物理机是Ubuntu16.04 + 1080 + CUDA 11.0，可以正常运行。模型来自于[SqueezeNet](https://github.com/forresti/SqueezeNet)

