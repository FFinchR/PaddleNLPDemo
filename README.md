# PaddleNLP demo仓库
## 参考文献：

### UIE框架(PaddleNLP)
[PaddleNLP信息抽取技术重磅升级！开放域信息抽取来了！三行代码用起来~](https://aistudio.baidu.com/aistudio/projectdetail/3914778?channelType=0&channel=0)

[win10 安装Paddlepaddle-GPU](https://aistudio.baidu.com/aistudio/projectdetail/3383520?channelType=0&channel=0)

[CUDA和cuDNN安装](https://blog.csdn.net/weixin_43082343/article/details/119043543)

[基于ERNIR3.0文本分类：CAIL2018-SMALL罪名预测为例(多标签)](https://aistudio.baidu.com/aistudio/projectdetail/4374631?channelType=0&channel=0)

[基于Ernie-3.0 CAIL2019法研杯要素识别多标签分类任务](https://aistudio.baidu.com/aistudio/projectdetail/4280922?contributionType=1)

[GitHub CAIL2022--司法文本信息抽取](https://github.com/china-ai-law-challenge/CAIL2022/tree/main/xxcq)

[上海市高级人民法院--裁判文书](http://www.hshfy.sh.cn/shfy/gweb2017/flws_list_new.jsp)

[官网教程--Windows下的PIP安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/windows-pip.html)

[官网教程--Linux下的PIP安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/linux-pip.html)

### Ubert框架(Fengshenbang-LM)

[Ubert: 统一 NLU 任务新范式](https://github.com/IDEA-CCNL/Fengshenbang-doc/blob/main/source/docs/%E4%BA%8C%E9%83%8E%E7%A5%9E%E7%B3%BB%E5%88%97/Erlangshen-Ubert-110M-Chinese.md)

[GitHub--Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)

### PaddleOCR + PaddleNLP应用
[使用PaddleNLP UIE模型抽取PDF版上市公司公告](https://aistudio.baidu.com/aistudio/projectdetail/4497591?channelType=0&channel=0)
[PaddleOCR 关键信息抽取](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/kie.md)

## 总结
### UIE信息抽取框架
UIE是PaddleNLP结合文心大模型中的知识增强NLP大模型ERNIE 3.0开源的首个面向通用信息抽取的产业级技术方案，不需要标注数据（或仅需少量标注数据），即可快速完成实体，关系，事件，情感等各类信息抽取任务。

该框架依托PaddleNLP，文档较为完备，且有实际案例可参考，小样本微调模型的抽取结果也略优于Ubert框架，但可能存在训练模型耗时较长的问题。
### Ubert信息抽取框架
Ubert 是2022AIWIN 世界人工智能创新大赛：中文保险小样本多任务A/B榜榜首的一种解决方案。相比于官方提供的 baseline，提高 20 个百分点。Ubert 不仅可以完成实体识别、事件抽取等常见抽取任务，还可以完成新闻分类、自然语言推理等分类任务，且所有任务是共享一个统一框架、统一任务、统一训练目标的模型。

但是由于是刚刚开源的框架，文档较少，几乎没有可参考的案例；并且由于此框架是针对于2022AIWIN开发的，该大赛不包含关系抽取任务，所以关系抽取的代码未开源，目前只能进行实体识别任务。
