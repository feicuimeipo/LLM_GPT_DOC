

# BERT

> tensorflow - 已不流行了, 现在主流的都是使用pytorch
>
> 虽然现在被替代，但是影响力巨大。



**Paper:** **Pre-training** of **Deep Birdirectional Transformers** for **Language Understanding**

**Name:** Birdrectional Encoder Representation from Transformers

---

**Pre-training**：通过大量的数据预训练得到的通用模型，后续基于通用模型进行微调。

**Deep**： BERT-base采用12层Encoder， 层数更深。

**Bidirectional**： 通过MLM任务实现上下文理解。

**Transformer**：基于Transformer的Encoder。

**Language Understanding**： BERT是一个语言模型，通过深层双向Encoder实现语义理解。

**Self-supervised Learning**: 通过设计合理的训练任务，实现使用大量无监督数据进行训练。

> GPT包括BERT开启了一个时代，进入了Pre-training阶段



## **BERT — 自监督学习**

<img src="19_学度学习的经典模型_Bert.assets/image-20231022013421515.png" alt="image-20231022013421515" style="zoom:67%;" />



## **BERT — 预训练任务MLM**

### **BERT — 预训练任务MLM**

![image-20231022013510552](19_学度学习的经典模型_Bert.assets/image-20231022013510552.png)

### **BERT — 预训练任务NSP**

![image-20231022013607891](19_学度学习的经典模型_Bert.assets/image-20231022013607891.png)



![image-20231022013614321](19_学度学习的经典模型_Bert.assets/image-20231022013614321.png)







## **BERT — input**

![image-20231022013713119](19_学度学习的经典模型_Bert.assets/image-20231022013713119.png)



## **BERT — Encoder**

![image-20231022013748108](19_学度学习的经典模型_Bert.assets/image-20231022013748108.png)



## **BERT — finetune**



![image-20231022013805558](19_学度学习的经典模型_Bert.assets/image-20231022013805558.png)



## RoBERTa

![image-20231022013844623](19_学度学习的经典模型_Bert.assets/image-20231022013844623.png)



![image-20231022013850077](19_学度学习的经典模型_Bert.assets/image-20231022013850077.png)



## **ALBERT**

![image-20231022013901288](19_学度学习的经典模型_Bert.assets/image-20231022013901288.png)



# T5 

![image-20231022013946357](19_学度学习的经典模型_Bert.assets/image-20231022013946357.png)





## Archietecture

![image-20231022013956394](19_学度学习的经典模型_Bert.assets/image-20231022013956394.png)

![image-20231022014003243](19_学度学习的经典模型_Bert.assets/image-20231022014003243.png)



![image-20231022014010281](19_学度学习的经典模型_Bert.assets/image-20231022014010281.png)



## **T5 — Objectives**

![image-20231022014030360](19_学度学习的经典模型_Bert.assets/image-20231022014030360.png)



![image-20231022014044463](19_学度学习的经典模型_Bert.assets/image-20231022014044463.png)



![image-20231022014055422](19_学度学习的经典模型_Bert.assets/image-20231022014055422.png)



![image-20231022014118020](19_学度学习的经典模型_Bert.assets/image-20231022014118020.png)



![image-20231022014124957](19_学度学习的经典模型_Bert.assets/image-20231022014124957.png)



## **T5 — Model**

![image-20231022014136575](19_学度学习的经典模型_Bert.assets/image-20231022014136575.png)





## **T5 — Performance**

![image-20231022014148518](19_学度学习的经典模型_Bert.assets/image-20231022014148518.png)





# 隐含词的理解能力

> 人民日报



# 附录

> 大模型做多轮对话的对话管理 —— 无法实现
>
> COT：组织训练
>
> 百度正在将文心一言：作为搜索的入口





### SOTA！模型

[197个经典SOTA模型](https://www.bilibili.com/read/cv26837375/)

