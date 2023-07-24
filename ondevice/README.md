**文件说明**

ondevice_searching_ea.py                使用遗传算法的搜索策略
ondevice_searching_sa.py                使用模拟退火算法的搜索策略
dynamic_adapt.py                              部署阶段，自适应环境变化
ValForDevice                                       边缘端数据集
npy/idxs.npy                                       边缘数据集的索引  
npy/subnets.npy                                不同的tradeoff下最优子结构的集合
pretrainedweight                               预训练模型
pths_for_device_torchvision            可单个加载的block集合

figure_11.sh 论文Figure 11对应的六组数据
figure_12.sh 论文Figure 12对应的六组数据
figure_13.sh 论文Figure 13对应的数据
table2.sh    论文Table2对应的数据（无CUDA version changed项）

注：
    边缘端数据集使用ImagetNet模拟
    subnets.npy由搜索阶段得到
    单个加载的block可由tools/mytools.extract_blocks_from_multimodel（）生成
