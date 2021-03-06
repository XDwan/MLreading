
[TOC]

# Comprehensive Privacy Analysis of Deep Learning

## 摘要

由于深度神经网络会记住训练数据的信息，所以很容易被不同的推理攻击入侵

## 简介

### 攻击类型

推理攻击主要分成两类：

- 成员推断
    在成员推断中，攻击者的目标是推断一个特别的个人数据记录是否包含在训练集中

- 重建攻击
    在重建攻击中，攻击者的目标是训练集中的记录的属性

**成员推断**，是一个决策的问题，模型的准确性直接影响模型能够泄漏多少训练数据
所以本文采用成员推理攻击

### 相同工作

最近的针对机器学习模型的成员推理攻击，通常在黑箱的设定下攻击者只能观察到模型的预测值

这些工作的结果就是展示训练数据的分布和模型的泛化能力，这些导致成员关系的泄漏。

过拟合的模型比泛化的模型更容易受到成员推断的攻击

然而这种黑盒攻击对于泛化能力很好的深度神经网络效果不是很好，但是在现实生活中，深度学习的算法的参数一般都是可见的

### 本文贡献

本文展现了一种综合能力很好的框架，能使用**白箱成员推断攻击**对深度神经网络进行隐私分析

且模型不一定要完全训练出来。当模型正在训练时，攻击者能被动的观察到模型的更新或者积极地影响目标模型使得泄漏更多的信息和不同类型的先验知识。

不管知识、观察还是攻击者的动作，最终目的只有一个：**成员推断** 

将现存的黑盒攻击简单拓展到白盒攻击一个简单的拓展是对模型所有激活函数使用相同的攻击，但这并不会提高精度

因此设计白盒攻击利用**随机梯度下降算法**中的隐私漏洞

在随机梯度下降算法中，每个数据样本都将对模型产生一点影响。因此要做的就是区分每个数据样本之间造成影响的差别

所以设计一个深度学习攻击模型，从目标模型中的不同层中提取特征并结合他们的信息来计算出目标数据样本的从属概率

通过实验，泛化能力较好的模型，黑盒攻击的效果较差，但是本文的白盒攻击的效果就较好。

## 推断攻击

### A. Attack Observations: Black-box vs. White-box Inference


![table1](Comprehensive%20privacyanalysis%20of%20deep%20learning/Table1.png)
![white](Comprehensive%20privacyanalysis%20of%20deep%20learning/whitebox.png)