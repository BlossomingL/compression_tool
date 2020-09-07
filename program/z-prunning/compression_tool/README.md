# compression_tool
A compression tool for minivision

## Install
git clone https://github.com/BlossomingL/Distiller  
git clone https://github.com/BlossomingL/compression_tool  

安装Distiller：在Distiller文件下，python setup.py install  

环境：pytorch 1.X  

## 程序文件列表和文件功能  
* commen_utils: 一些常用的工具方法  
* commen_utils: 一些常用的工具方法
* data: 存放测试数据集和训练数据集，以及需要的pair list
* model_define: 存放待剪枝的模型定义文件
* pruning_analysis_tools: 两个剪枝分析工具，auto_make_yaml.py根据得到的csv文件自动生成yaml配置文件,plot_csv.py根据csv文件画出敏感度折线图。
* test_module:  存放不同的测试模块（计算精度，ROC等）
* train_module: 存放不同的训练模块
* work_space: 存放各种模型文件，模型定义文件，以及敏感度分析数据(csv文件)
* yaml_file: 存放自动生成的yaml文件
* dataloader.py: 数据多线程批加载
* main.py: 主程序入口
* prune.sh: 剪枝脚本
* pruning.py: 剪枝相关代码
* quantization.py:  量化相关代码
* quantization.sh: 量化运行脚本
* sensitivity_analysis.py: 敏感度分析，生成csv文件

## 剪枝运行流程  
* 准备阶段：本工具运行需要安装distiller，安装文件位于Distiller，在Distiller目录下打开命令行，运行python setup.py install。准备模型定义文件放入model_define文件下，准备训练最高精度的pt文件放在work_space/model_train_best下，准备测试集放在data下以及测试集测试代码放在test_module下。

* 敏感度分析: 运行pruning.py文件，例如对resnet100进行剪枝敏感度分析，sh脚本如下

```python
python pruning.py --mode sa \         # sa(sensitivity analysis)表示进入敏感度分析模式
                  --model resnet100 \  # 模型名称，只能从固定的几个选择
                  --best_model_path work_space/model_train_best/2019-09-29-11-37_SVGArcFace-O1-b0.4s40t1.1_fc_0.4_112x112_2019-09-27-Adult-padSY-Bus_fResNet100v3cv-d512_model_iter-340000.pth \   # 训练好的模型文件
                  --from_data_parallel \  # 上面的模型文件是否是多卡训练得到
                  --test_root_path data/test_data/fc_0.4_112x112 \ # 测试集root路径
                  --img_list_label_path   # pair list路径 data/test_data/fc_0.4_112x112/pair_list/id_life_image_list_bmppair.txt \
                  --fpgm \ # 采用fpgm算法剪枝
                  --data_source company # 数据集来源
```
运行过后会生成一个csv文件，在work_space/sensitivity_data下
* 生成yaml文件: 运行pruning_analysis_tools下的auto_make_yaml.py文件，其中config_yaml函数的参数需要自己配置，参数1：csv文件路径，参数2：期望剪枝后的精度，参数3：模型名称，比如上述resnet100,那么此参数为resnet100，参数4: 输入图像大小。运行后会在yaml_file文件夹下生成一个yaml文件。

* 剪枝：运行pruning.py文件，例如对resnet100进行剪枝，sh脚本如下
```python
python pruning.py --mode prune \
                  --model resnet100 \
                  --best_model_path work_space/model_train_best/2019-09-29-11-37_SVGArcFace-O1-b0.4s40t1.1_fc_0.4_112x112_2019-09-27-Adult-padSY-Bus_fResNet100v3cv-d512_model_iter-340000.pth \
                  --from_data_parallel \
                  --fpgm \
                  --save_model_pt \    # 是否保存剪枝后的模型文件
                  --test_root_path data/test_data/fc_0.4_112x112 \
                  --img_list_label_path data/test_data/fc_0.4_112x112/pair_list/id_life_image_list_bmppair.txt \
                  --data_source company
```
运行后会生成一个剪枝后的模型文件，保存在work_space/pruned_model，文件名与模型名称一致，并且还会打印出每层的out参数，此参数即剪枝后模型定义文件中的keep数组。  

## 关于distiller  
由于此工具硬剪枝部分(即真正将通道移除)的代码是采用distiller框架中的代码，因为模型的特殊，需要更改框架源码才能进行剪枝，下面对更改的部分说明：
* distiller/apputils下的data_loaders.py文件classification_get_input_shape函数中dataset与yaml文件中dataset参数的值一样，例如：如果输入网络图像大小为80x80那么yaml文件中的dataset参数也就是80x80，如果需要添加其它类型的输入，那么就要更改此代码。
* distiller/policy.py下添加了fpgm的参数选项
* distiller/thinning.py下添加了两个个功能：
    * 能够剪PReLU层，具体函数为handle_prelu_layers，append_prelu_thinning_directive(注：如果PReLU层采用默认的参数1，那么需要将此代码注释掉，否则会出错)
    * 针对公司Block的第一层为BN层，源代码本身不支持对此层剪枝，更改后可支持。具体函数为handle_bn_layers_bn1。
* distiller/pruning/ranked_structures_pruner.py下添加了fpgm算法。具体代码为rank_and_prune_filters函数中if fpgm开始到if结束。
* distiller/summary_graph下更改源码一处BUG，在add_footprint_attr函数下加入try/catch模块
* 增加CVPR 2020 HRank剪枝方法  

## 参考
[1] HRank：https://arxiv.org/abs/2002.10179  
[2] FPGM: https://arxiv.org/abs/1811.00250  
[3] Distiller: https://github.com/NervanaSystems/distiller
