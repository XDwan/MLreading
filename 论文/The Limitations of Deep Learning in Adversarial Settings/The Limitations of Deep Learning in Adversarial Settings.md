[TOC]

# The Limitations of Deep Learning in Adversarial Settings

## Abstract 

深度学习利用 **大量数据和高效的算法** 展现出比别的机器学习任务更好的性能

但是训练阶段的缺点导致深度神经网络更容易受到对抗样本的攻击： 攻击者制作输入使得DNN误判

1. 针对DNN formalize the space of adversaries
2. 基于已完全了解的DNN的输入与输出的映射，提出了一种洗的算法来制作对抗样本
3. 在计算机视觉的应用中，展现 对抗样本 能被人类识别但无法被机器识别的概率高达97%
4. 通过定义一种 hardness measure 对不同样本类型进行对抗扰动的脆弱性进行了评估
5. 通过设定正常输入和目标分类之间的预测距离 进行初步的防御


## Introduction

**对抗样本** ：是一个特制的输入 能导致学习算法误判

利用DNN从有限训练集中学习到的不完全泛化，通过添加扰动，以及用于构建DNN中使用的大多数元件都有潜在的线度，从良性样本中创建对抗性样本

多维函数 $F:X->Y$ 其中$X$是一个原始特征向量，$Y$是输出数组
然后通过加上一个$\delta X$构建一个$X^*$ 

$arg \; \underset{{\delta X} }{\min}\;\rVert \delta X \rVert \;s.t. F(X+\delta X)=Y^* \qquad(1)$

其中$X^* \; = X +\delta X$
解决这个问题很重要，因为DNN通常是非线性和非凸

我们对输入的变化如何影响深度神经网络输出的理解源于前向导数（forward derivative）：我们引入并定义为DNN学习的函数的Jacobian矩阵。

为了生成能从DNN模型中获得理想结果的对抗样本，就用前向导数构造 包含扰动$\delta X$的对抗显著映射（adversarial saliency maps）

基于前向导数的方法比在先验系统中使用的梯度下降技术要强大得多：

- 适用于监督和无监督学习
- 能为广泛对抗样本族类生成信息

对抗显著性映射是基于前向导数的一个重要工具，并在设计时考虑到对抗性目标，在扰动的选择方面给予敌手更大的控制权

## Contribution

1. 在对抗性目标和能力方面，我们形式化了对手对分类器DNN的空间
2. 我们引入了一类新的算法，仅通过使用DNN架构的知识来制作对抗性样本
3. 我们使用一个广泛使用的计算机视觉DNN来验证了这些算法。

## Threat Model Taxonomy in Deep Learning（深度学习中的威胁模型分类）

![图 1](../../images/cae27932abca299433f0ebdc685ded3c8933ea2d178a959ee8c88bad4d8d8c64.png)  

### About Deep Neural Networks

深度神经网络是一种组织成神经元层的大型神经网络，对应于输入数据的连续表示。

DNN 又有两类 **监督**与**无监督**

- 监督训练导致使用从标记训练数据推断的函数将未见样本映射到预定义的输出集的模型
- 无监督训练学习未标记训练数据，由此产生的DNN模型可以用于生成新的样本，或者通过作为大型DNN的预处理层来自动化特征工程

![图 2](../../images/6702a05d2b3c94afcbd012c4d45d5131da8eb58d27b65ef774d73b02aa4acfea.png) 
$$\phi:X ->\dfrac{1}{1+e^{-x}} $$

$$Z_{h_1}(X) = \omega_{11}x_1+\omega_{12}x_2+b_1$$

$$h_1(X)=\phi(Z_{h_1}(X))$$

$$Z_{0}(X) = \omega_{31}h_1(X)+\omega_{32}h_2(X)+b_3$$

### Adversarial Goals（敌对目标）

1. Confidence reduction 降低输出置信度
2. Misclassification 将输出类改为与原类不同的类
3. Targeted misclassification 产生能强迫输出成为一个特定的目标类的输入
4. Source/target misclassification 强迫一个特定输入的输出分类导向一个特定的目标类

### Adversarial Capabilities（敌对能力）

敌手由其掌握的信息和能力定义

1. Training data and network architecture 敌手全知模型和训练数据
2. Network architecture 敌手知道网络架构及其参数值
3. Training data 敌手能获得训练集同分布的一个子集
4. Oracle 敌手能使用神经网络（能获得其输出）
5. Samples 敌手能获得神经网络的输入输出对

## Approach

提出了一个修改样本的通用算法，使DNN产生任何期望的敌对输出。

对模型结构及其参数的知识足够获得针对acyclic feedforward DNNs的敌对样本

这需要评估DNN的前向导数，以构建一个对抗性显著性映射，以识别与对手目标相关的输入特征集。

扰乱以这种方式识别的特征会迅速导致所需的对抗性输出，例如，错误的分类

虽然我们用监督分布式神经网络作为分类器来描述我们的方法，但它也适用于无监督架构。

### Studying a Simple Neural Network

上面对模型结构的阐述，展示了使用前向导数发现的小输入扰动是如何引起神经网络输出的大变化的。

输入偏差为$b_1,b_2,b_3(null)$

应得输出为$F(X) = x_1 \bigwedge x_2$ 其中$X = (x_1,x_2)$ 非整数会化成整数

这些扰动可以通过优化或者手动解决，但这些解决方法不适用与DNN，因其通常是非凸与非线性，因此提出一种基于向前导数的算法

$J_{X}=[\dfrac{\partial F(X)}{\partial x_1},\dfrac{\partial F(X)}{\partial x_2}]$