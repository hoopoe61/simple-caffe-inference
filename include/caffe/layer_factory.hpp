/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_HPP_
#define CAFFE_LAYER_FACTORY_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

/**********************************
 * 此段代码的逻辑为:
 * 1.每个层文件最后会调用宏REGISTER_LAYER_CLASS,来进行注册;
 * 2.在宏REGISTER_LAYER_CLASS中,会调用宏REGISTER_LAYER_CREATOR,来创建creator;
 * 3.该creator会实例化一个类LayerRegisterer,调用LayerRegistry中的AddCreator函数来加入到一个
 *    字典中;
 * 4.在注册完后,之后的使用都是调用类LayerRegistry中的CreateLayer函数
**********************************/

/*********************************
 * 注意:这里有两种方式来注册层:
 * 1.利用宏REGISTER_LAYER_CLASS来进行注册;
 * 2.利用宏REGISTER_LAYER_CREATOR来进行注册.
 * 两者区别在于是否具有creator.建议使用REGISTER_LAYER_CLASS来进行注册.
**********************************/

namespace caffe
{
    template <typename Dtype>
    class Layer;

    template <typename Dtype>
    class LayerRegistry
    {
    public:
        typedef shared_ptr<Layer<Dtype>> (*Creator)(const LayerParameter &);
        typedef std::map<string, Creator> CreatorRegistry; //key:层名字,value:注册器

        static CreatorRegistry &Registry()
        {
            //TODO 哪里delete?不会内存泄露吗
            static CreatorRegistry *g_registry_ = new CreatorRegistry();
            return *g_registry_;
        }

        //增加creator
        static void AddCreator(const string &type, Creator creator)
        {
            CreatorRegistry &registry = Registry();
            //判断是否已经注册过
            CHECK_EQ(registry.count(type), 0)
                << "Layer type " << type << " already registered.";
            registry[type] = creator;
        }

        //使用LayerParameter来初始化层
        static shared_ptr<Layer<Dtype>> CreateLayer(const LayerParameter &param)
        {
            // if (Caffe::root_solver()) {
            //   LOG(INFO) << "Creating layer " << param.name();
            // }
            const string &type = param.type();
            CreatorRegistry &registry = Registry();
            //判断已经注册过
            CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type;
            return registry[type](param); //使用LayerParameter参数来初始化该层
        }

    private:
        // Layer registry should never be instantiated - everything is done with its
        // static variables.
        LayerRegistry() {}

    }; //class LayerRegistry

    template <typename Dtype>
    class LayerRegisterer
    {
    public:
        LayerRegisterer(const string &type,
                        shared_ptr<Layer<Dtype>> (*creator)(const LayerParameter &))
        {
            // LOG(INFO) << "Registering layer type: " << type;
            LayerRegistry<Dtype>::AddCreator(type, creator);
        }
    }; //class LayerRegisterer

//如果存在creator,则直接调用该宏
#define REGISTER_LAYER_CREATOR(type, creator)                                \
    static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>); \
    static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)

// 每个层文件最后会调用该宏定义,然后会将相应的层进行注册
// 如果没有定义creator,则会调用初始化函数来作为creator
#define REGISTER_LAYER_CLASS(type)                                              \
    template <typename Dtype>                                                   \
    shared_ptr<Layer<Dtype>> Creator_##type##Layer(const LayerParameter &param) \
    {                                                                           \
        return shared_ptr<Layer<Dtype>>(new type##Layer<Dtype>(param));         \
    }                                                                           \
    REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

} // namespace caffe

#endif //CAFFE_LAYER_FACTORY_HPP_