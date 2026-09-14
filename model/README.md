# Model来源与分割适配

原项目：https://github.com/shuaihao-han/MODA
论文：Han, Shuaihao; Xu, Tingfa; Liu, Peifu; Li, Jianan. MODA: The First Challenging Benchmark for Multispectral Object Detection in Aerial Images. AAAI 2026.

cfb.py、ssaf.py、fem.py从上游tools/my_module/原样复制。随附LICENSE.MODA为上游根目录Apache-2.0许可证；未复制MODA数据或权重。上游源码自己的注释有spectral等词汇，在本RGB版本中这些模块实际处理学习到的通道特征，不代表原始多光谱波段。

| 部分 | 改造 |
|---|---|
| 八波段ResNet-50 | 替换为torchvision RGB ResNet-50 |
| 横向投影 | 用GroupNorm卷积重写 |
| CFB、SSAF、FEM | 保留上游源码和主要前向连接顺序；在AMP中用FP32执行 |
| 框生成的目标图、ActivationLoss | 改为真实植物像素掩膜的BCE监督 |
| 前景引导 | 保留F*(1+sigmoid(aux))残差加权 |
| 旋转框头、额外检测尺度、NMS | 移除 |
| 分割输出 | 新增多尺度解码器、半分辨率RGB细节分支、两类别像素预测 |
| 主损失 | 交叉熵与前景Dice，辅助BCE权重0.2 |

新适配代码不是作者的官方分割实现；不声称原检测实验复现。所有模型结构与损失在本目录；训练循环只接收接口。新增适配代码按Apache-2.0提供；仓库中原模板文件保留其原有权利归属。发表实验时引用MODA，并明确哪些模块沿用、哪些结构为分割适配。

上游Git blob SHA（2026-09-14读取）：

- cfb.py: 71715072354fbec340ebb566faa7828524ee3d71
- ssaf.py: 7fcbf7e42e8172bf64558ab1bdfedae8ab1feee2
- fem.py: 37dbb5316654f9edf5d3bbcb31f3c658ade7a767
