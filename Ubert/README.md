# Fengshenbang-LM UbertPipeline 信息抽取

# 1. 框架安装

安装前，检查Python版本，需≥3.8，且为 64位版本，并将pip升级到最新版本。

```bash
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
pip install --editable ./ --use-deprecated=legacy-resolver
```

clone过程中可能会提示 `Permission denied`，这将导致 `Fengshenbang-LM/fengshen/data/fs_datasets` 目录为空，解决方法是在 `Fengshenbang-LM/fengshen/data` 目录下执行：

```bash
git clone https://github.com/IDEA-CCNL/fs_datasets.git
```

# 2. 开箱即用

一键运行下面代码得到预测结果, 你可以任意修改示例 text 和要抽取的 entity_type，体验一下 Zero-Shot 性能

```python
import argparse
from fengshen import UbertPiplines

total_parser = argparse.ArgumentParser("TASK NAME")
total_parser = UbertPiplines.piplines_args(total_parser)
args = total_parser.parse_args()
args.pretrained_model_path = 'IDEA-CCNL/Erlangshen-Ubert-110M-Chinese'  #预训练模型路径
test_data=[
    {
        "task_type": "抽取任务", 
        "subtask_type": "实体识别", 
        "text": "这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。", 
        "choices": [ 
            {"entity_type": "小区名字"}, 
            {"entity_type": "岗位职责"}
            ],
        "id": 0}
]

model = UbertPiplines(args)
result = model.predict(test_data)
for line in result:
    print(line)
```

下面是预测结果：

```
{'task_type': '抽取任务', 'subtask_type': '实体识别', 'text': '这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。', 'choices': [{'entity_type': '小区名字', 'entity_list': [{'entity_name': '雅清苑', 'score': 0.9819884173485227}]}, {'entity_type': '岗位职责', 'entity_list': [{'entity_name': '公务员', 'score': 0.759721709453616}]}], 'id': 0}
```

# 3. Ubert与UIE Zero-Shot性能对比
## 3.1 测试数据

选择Ubert与UIE提供的简单的实体抽取数据，进行Zero-Shot性能对比。

Ubert提供的数据：

```
这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。
schema = ['小区名字', '岗位职责']
```

UIE提供的数据

```
2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！
schema = ['时间', '选手', '赛事名称']

（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。
schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
```

法律判决书数据

```text
"上海金融法院"
"民事判决书"
"案号：（2022）沪74民终186号"
"上诉人（原审被告）：潢川县发展投资有限责任公司，住所地河南省信阳市潢川县财政局。"
"法定代表人：梅利平，董事长。"
"委托诉讼代理人：李明修，该公司员工。"
"委托诉讼代理人：李阳，河南捷达律师事务所律师。"
"被上诉人（原审原告）：海发宝诚融资租赁有限公司（原名中远海运租赁有限公司），住所地中国（上海）自由贸易试验区福山路450号3E室。"
"法定代表人：陈易明，总经理。"
"委托诉讼代理人：邹华恩，北京德和衡（上海）律师事务所律师。"
"委托诉讼代理人：王坤，北京德和衡（上海）律师事务所律师。"
```

## 3.2 Ubert预测结果

Ubert的预测结果如下：

```
{'choices': [{'entity_list': [{'entity_name': '雅清苑',
                               'score': 0.9367575436302099}],
              'entity_type': '小区名字'},
             {'entity_list': [{'entity_name': '公务员',
                               'score': 0.6243673748128286}],
              'entity_type': '岗位职责'}],
 'id': 0,
 'subtask_type': '实体识别',
 'task_type': '抽取任务',
 'text': '这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。'}

{'choices': [{'entity_list': [], 'entity_type': '时间'},
             {'entity_list': [{'entity_name': '北京冬奥会自由式滑雪女子大跳台决赛',
                               'score': 0.5273578842020762}],
              'entity_type': '赛事名称'},
             {'entity_list': [{'entity_name': '谷爱凌',
                               'score': 0.9639831676998543}],
              'entity_type': '选手'}],
 'id': 1,
 'subtask_type': '实体识别',
 'task_type': '抽取任务',
 'text': '2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！'}

{'choices': [{'entity_list': [{'entity_name': '4.2×4.0×2.8cm',
                               'score': 0.8498448066471739}],
              'entity_type': '肿瘤的大小'},
             {'entity_list': [{'entity_name': '1个',
                               'score': 0.8163116941904662}],
              'entity_type': '肿瘤的个数'},
             {'entity_list': [{'entity_name': 'II-III级',
                               'score': 0.7418857249761116}],
              'entity_type': '肝癌级别'},
             {'entity_list': [{'entity_name': 'M0级',
                               'score': 0.6987325687284445}],
              'entity_type': '脉管内癌栓分级'}],
 'id': 2,
 'subtask_type': '实体识别',
 'task_type': '抽取任务',
 'text': '（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。'}

{'choices': [{'entity_list': [{'entity_name': '上海金融法院',
                               'score': 0.9990102552582323}],
              'entity_type': '法院'},
             {'entity_list': [], 'entity_type': '原告'},
             {'entity_list': [{'entity_name': '梅利平',
                               'score': 0.6782786206834687},
                              {'entity_name': '李明修',
                               'score': 0.9517110008301973},
                              {'entity_name': '李阳',
                               'score': 0.9133601218681332},
                              {'entity_name': '陈易明',
                               'score': 0.601037533085888},
                              {'entity_name': '邹华恩',
                               'score': 0.9616709439037242},
                              {'entity_name': '王坤',
                               'score': 0.9794953910465246}],
              'entity_type': '原告委托代理人'},
             {'entity_list': [{'entity_name': '潢川县发展投资有限责任公司',
                               'score': 0.939025092158108},
                              {'entity_name': '海发宝诚融资租赁有限公司',
                               'score': 0.9325847653992343},
                              {'entity_name': '中远海运租赁有限公司',
                               'score': 0.5655144816017525}],
              'entity_type': '被告'},
             {'entity_list': [{'entity_name': '李明修',
                               'score': 0.9318551587316268},
                              {'entity_name': '李阳',
                               'score': 0.8713249213930339},
                              {'entity_name': '邹华恩',
                               'score': 0.9507123267095962},
                              {'entity_name': '王坤',
                               'score': 0.9769380127811919}],
              'entity_type': '被告委托代理人'}],
 'id': 3,
 'subtask_type': '实体识别',
 'task_type': '抽取任务',
 'text': '上海金融法院民事判决书案号：（2022）沪74民终186号上诉人（原审被告）：潢川县发展投资有限责任公司，住所地河南省信阳市潢川县财政局。法定代表人：梅利平，董事长。委托诉讼代理人：李明修，该公司员工。委托诉讼代理人：李阳，河南捷达律师事务所律师。被上诉人（原审原告）：海发宝诚融资租赁有限公司（原名中远海运租赁有限公司），住所地中国（上海）自由贸易试验区福山路450号3E室。法定代表人：陈易明，总经理。委托诉讼代理人：邹华恩，北京德和衡（上海）律师事务所律师。委托诉讼代理人：王坤，北京德和衡（上海）律师事务所律师。'}
```
Ubert在自己提供的示例中表现很好，但在抽取UIE提供的测试数据时存在一些问题，`'2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！'`
，此条数据未抽取到`时间`实体；在法律判决书数据上，表现糟糕，几乎没有正确标注。

## 3.3 UIE预测结果
以下是UIE的预测结果：

```
[{'小区名字': [{'end': 15,
            'probability': 0.8576971485350953,
            'start': 12,
            'text': '雅清苑'}]}]

[{'时间': [{'end': 6,
          'probability': 0.9857378532924486,
          'start': 0,
          'text': '2月8日上午'}],
  '赛事名称': [{'end': 23,
            'probability': 0.8503089953268272,
            'start': 6,
            'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
  '选手': [{'end': 31,
          'probability': 0.8981548639781138,
          'start': 28,
          'text': '谷爱凌'}]}]

[{'肝癌级别': [{'end': 20,
            'probability': 0.9243267447402701,
            'start': 13,
            'text': 'II-III级'}],
  '肿瘤的个数': [{'end': 84,
            'probability': 0.7538413804059623,
            'start': 82,
            'text': '1个'}],
  '肿瘤的大小': [{'end': 100,
            'probability': 0.8341128043459491,
            'start': 87,
            'text': '4.2×4.0×2.8cm'}],
  '脉管内癌栓分级': [{'end': 70,
              'probability': 0.9083292325934664,
              'start': 67,
              'text': 'M0级'}]}]
              
[{'原告': [{'end': 52,
      'probability': 0.48330004363117496,
      'relations': {'委托代理人': [{'end': 94,
                               'probability': 0.6823175581385357,
                               'start': 91,
                               'text': '李明修'}]},
      'start': 39,
      'text': '潢川县发展投资有限责任公司'}],
'法院': [{'end': 6,
      'probability': 0.9861440082803696,
      'start': 0,
      'text': '上海金融法院'}],
'被告': [{'end': 52,
      'probability': 0.5244898578090975,
      'relations': {'委托代理人': [{'end': 94,
                               'probability': 0.6823175581385357,
                               'start': 91,
                               'text': '李明修'}]},
      'start': 39,
      'text': '潢川县发展投资有限责任公司'}]}]   
```
同样的，UIE在自己提供的测试数据中表现很好，但在预测Ubert提供的数据时，无法正确标注`岗位职责`；在抽取法律判决书数据时，同样存在一定问题，如原告、被告识别反了，委托代理人识别错误等。

## 3.4 小结
从预测结果可知，Ubert与UIE在各自提供的示例中均表现得很好；

而在使用Zero-Shot预测对方示例时，均存在一些问题；

在标注稍复杂的法律判决书数据时，表现都不尽人意。

# 4. finetune使用

对细分场景继续finetune,参考 [example.py](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/ubert/example.py)，需要将数据预处理成指定的格式。

该训练复用了 pytorch-lightning 的 trainer 。在训练时，可以直接传入 trainer 的参数。

此外还定义了一些其他参数。常用的参数如下：

```
--pretrained_model_path       #预训练模型的路径，默认
--load_checkpoints_path       #加载模型的路径，如果你finetune完，想加载模型进行预测可以传入这个参数
--batchsize                   #批次大小, 默认 8
--monitor                     #保存模型需要监控的变量，例如我们可监控 val_span_acc
--checkpoint_path             #模型保存的路径, 默认 ./checkpoint
--save_top_k                  #最多保存几个模型, 默认 3
--every_n_train_steps         #多少步保存一次模型, 默认 100
--learning_rate               #学习率, 默认 2e-5
--warmup                      #预热的概率, 默认 0.01
--default_root_dir            #模型日子默认输出路径
--gradient_clip_val           #梯度截断， 默认 0.25
--gpus                        #gpu 的数量
--check_val_every_n_epoch     #多少次验证一次， 默认 100
--max_epochs                  #多少个 epochs， 默认 5
--max_length                  #句子最大长度， 默认 512
--num_labels                  #训练每条样本最多取多少个label，超过则进行随机采样负样本， 默认 10
```

```shell
python finetune.py --batch 1
```
以下是训练轮次为500轮次，finetune后的模型的抽取结果：
```text
{'choices': [{'entity_list': [{'entity_name': '上海金融法院',
                               'score': 0.9999988434740585}],
              'entity_type': '法院'},
             {'entity_list': [{'entity_name': '海发宝诚融资租赁有限公司（',
                               'score': 0.9335503554681707},
                              {'entity_name': '海发宝诚融资租赁有限公司（原名中远海运租赁有限公司），',
                               'score': 0.7926754172407782}],
              'entity_type': '原告'},
             {'entity_list': [{'entity_name': '邹华恩，',
                               'score': 0.9999987154828944},
                              {'entity_name': '王坤，',
                               'score': 0.99999362037407}],
              'entity_type': '原告委托代理人'},
             {'entity_list': [{'entity_name': '潢川县发展投资有限责任公司，',
                               'score': 0.9999978690812292}],
              'entity_type': '被告'},
             {'entity_list': [{'entity_name': '李明修，',
                               'score': 0.995225242199723},
                              {'entity_name': '李阳，',
                               'score': 0.999994859682157}],
              'entity_type': '被告委托代理人'}],
 'id': 0,
 'subtask_type': '实体识别',
 'task_type': '抽取任务',
 'text': '上海金融法院民事判决书案号：（2022）沪74民终186号上诉人（原审被告）：潢川县发展投资有限责任公司，住所地河南省信阳市潢川县财政局。法定代表人：梅利平，董事长。委托诉讼代理人：李明修，该公司员工。委托诉讼代理人：李阳，河南捷达律师事务所律师。被上诉人（原审原告）：海发宝诚融资租赁有限公司（原名中远海运租赁有限公司），住所地中国（上海）自由贸易试验区福山路450号3E室。法定代表人：陈易明，总经理。委托诉讼代理人：邹华恩，北京德和衡（上海）律师事务所律师。委托诉讼代理人：王坤，北京德和衡（上海）律师事务所律师。'}
```
可以看出，相比零样本，除抽取结果包含后缀标点符号的问题外，抽取结果也是正确的。
