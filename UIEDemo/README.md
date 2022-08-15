# PaddleNLP信息抽取uie-demo
## Content
PaddleNLP Taskflow UIE（通用信息抽取）案例预研与搭建
1. 完成Taskflow UIE工具的Windows与Linux环境搭建。
2. 使用doccano标注工具完成法律判决书数据的标注与导出。
3. 改造工具类，使其适配涉毒类法律文书数据格式（NYT数据集格式）
4. 在法律判决书场景与涉毒类法律文书场景中使用UIE工具完成零样本抽取，结果表示在细分场景下，零样本抽取效果有限。
5. 通过小样本训练，微调UIE模型，提升抽取效果。

法律场景-判决书抽取

![image](https://user-images.githubusercontent.com/40840292/169017863-442c50f1-bfd4-47d0-8d95-8b1d53cfba3c.png)
## PaddleNLP安装
```shell
pip install paddlenlp
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```
注意：PaddleNLP需要PyArrow支持，而PyArrow需要安装64位Python，建议使用Python3.7 64位版本。

## 信息抽取

以下每一小节均分为两部分，一部分是参考PaddleNLP给出的案例，抽取判决书的信息，包括：
```text
['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]
```
另一部分是使用PaddleNLP尝试进行[CAIL(CHina AI & Law Challenge)2022--司法文本信息抽取](https://github.com/china-ai-law-challenge/CAIL2022/tree/main/xxcq)分赛道，抽取信息包括：

```text
['地点', '时间', '毒品重量', {'人名': '人名'}, {'人名': '毒品类型'}, {'人名': '人名'}]
```
### 1.开箱即用
#### 判决书场景
- 在判决书场景同时对文本进行实体抽取和关系抽取，schema可按照如下方式进行构造：
    ```text
    ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]
    ```
    #### 调用示例：
    
    PaddleNLP提供了一个示例，如下：
    ```python
    >>> schema = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]
    >>> ie.set_schema(schema)
    >>> pprint(ie("北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。")) # Better print results using pprint
        [{'原告': [{'end': 37,
                  'probability': 0.9949814024296764,
                  'relations': {'委托代理人': [{'end': 46,
                                          'probability': 0.7956844697990384,
                                          'start': 44,
                                          'text': '李四'}]},
                  'start': 35,
                  'text': '张三'}],
          '法院': [{'end': 10,
                  'probability': 0.9221074192336651,
                  'start': 0,
                  'text': '北京市海淀区人民法院'}],
          '被告': [{'end': 67,
                  'probability': 0.8437349536631089,
                  'relations': {'委托代理人': [{'end': 92,
                                          'probability': 0.7267121388225029,
                                          'start': 90,
                                          'text': '赵六'}]},
                  'start': 64,
                  'text': 'B公司'}]}]
    ```
    可以发现，对于简单的文本，Taskflow表现尚可，接下来，我们使用稍复杂的判决书文本进行尝试：

    ```python
    >>> schema = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}] # Define the schema for entity extraction
    >>> ie = Taskflow('information_extraction', schema=schema)
    >>> pprint(ie("上海金融法院"
              "民事判决书"
              "案号：（2022）沪74民终186号"
              "上诉人（原审被告）：潢川县发展投资有限责任公司，住所地河南省信阳市潢川县财政局。"
              "法定代表人：梅利平，董事长。"
              "委托诉讼代理人：李明修，该公司员工。"
              "委托诉讼代理人：李阳，河南捷达律师事务所律师。"
              "被上诉人（原审原告）：海发宝诚融资租赁有限公司（原名中远海运租赁有限公司），住所地中国（上海）自由贸易试验区福山路450号3E室。"
              "法定代表人：陈易明，总经理。"
              "委托诉讼代理人：邹华恩，北京德和衡（上海）律师事务所律师。"
              "委托诉讼代理人：王坤，北京德和衡（上海）律师事务所律师。"))  # Better print results using pprint
    
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
    可以发现，抽取的结果存在一定问题，如原告、被告识别反了，委托代理人识别错误等。

#### 涉毒类法律文书场景
- 对另一个场景，数据集来自CAIL2022--司法文本信息抽取，主要内容来自于网络公开的若干涉毒类罪名法律文书，预定义了5种实体和4种关系类型，实体分别为`人名实体(Nh)、地名实体(Ns)、时间实体(NT)、毒品类型实体(NDR)、和毒品重量实体(NW)`，关系分别为`贩卖给（人）( sell_drug_to )，贩卖（毒品）( traffic_in )，持有( possess )，非法容留( provide_shelter_for )`。

    ```python
    >>> schema = ['地点', '时间', '毒品重量',  {'涉案人': '涉案人'}, {'涉案人': '毒品类型'}, {'涉案人': '涉案人'}]  # Define the schema for entity extraction
    >>> ie = Taskflow('information_extraction', schema=schema)
    >>> pprint(ie("2016年6月初某日，被告人陆xx经与吸毒人员范xx事先联系，由范xx驾驶牌号为沪C8XXXX红色奥迪轿车至本区卫清东路XXX号门口，被告人陆xx进入该车内将约0.3克甲基苯丙胺以人民币300元的价格贩卖给范xx"))  # Better print results using pprint
  [{'地点': [{'end': 66,
          'probability': 0.6405238209206026,
          'start': 54,
          'text': '本区卫清东路XXX号门口'}],
  '时间': [{'end': 10,
          'probability': 0.9186600131619826,
          'start': 0,
          'text': '2016年6月初某日'}]}]
    ```
    修改部分schema后：
    ```python
    >>> schema = ['地点', '时间', '毒品重量',  {'人名': '人名'}, {'人名': '毒品'}, {'人名': '人名'}]  # Define the schema for entity extraction
    >>> ie = Taskflow('information_extraction', schema=schema)
    >>> pprint(ie("2016年6月初某日，被告人陆xx经与吸毒人员范xx事先联系，由范xx驾驶牌号为沪C8XXXX红色奥迪轿车至本区卫清东路XXX号门口，被告人陆xx进入该车内将约0.3克甲基苯丙胺以人民币300元的价格贩卖给范xx"))  # Better print results using pprint
    [{'人名': [{'end': 104,
              'probability': 0.7289157282719678,
              'relations': {'毒品': [{'end': 89,
                                    'probability': 0.4290085292110035,
                                    'start': 84,
                                    'text': '甲基苯丙胺'}]},
              'start': 103,
              'text': '范'},
             {'end': 15,
              'probability': 0.792671543603717,
              'relations': {'毒品': [{'end': 89,
                                    'probability': 0.399578059593523,
                                    'start': 84,
                                    'text': '甲基苯丙胺'}]},
              'start': 14,
              'text': '陆'},
             {'end': 71,
              'probability': 0.8625131077428243,
              'relations': {'毒品': [{'end': 89,
                                    'probability': 0.399578059593523,
                                    'start': 84,
                                    'text': '甲基苯丙胺'}]},
              'start': 70,
              'text': '陆'},
             {'end': 24,
              'probability': 0.7906338199907594,
              'relations': {'毒品': [{'end': 89,
                                    'probability': 0.4290085292110035,
                                    'start': 84,
                                    'text': '甲基苯丙胺'}]},
              'start': 23,
              'text': '范'},
             {'end': 33,
              'probability': 0.7101535535871051,
              'relations': {'毒品': [{'end': 89,
                                    'probability': 0.4290085292110035,
                                    'start': 84,
                                    'text': '甲基苯丙胺'}]},
              'start': 32,
              'text': '范'},
             {'end': 104,
              'probability': 0.7289157282719678,
              'start': 103,
              'text': '范'},
             {'end': 15,
              'probability': 0.792671543603717,
              'start': 14,
              'text': '陆'},
             {'end': 71,
              'probability': 0.8625131077428243,
              'start': 70,
              'text': '陆'},
             {'end': 24,
              'probability': 0.7906338199907594,
              'start': 23,
              'text': '范'},
             {'end': 33,
              'probability': 0.7101535535871051,
              'start': 32,
              'text': '范'},
             {'end': 104,
              'probability': 0.7289157282719678,
              'start': 103,
              'text': '范'},
             {'end': 15,
              'probability': 0.792671543603717,
              'start': 14,
              'text': '陆'},
             {'end': 71,
              'probability': 0.8625131077428243,
              'start': 70,
              'text': '陆'},
             {'end': 24,
              'probability': 0.7906338199907594,
              'start': 23,
              'text': '范'},
             {'end': 33,
              'probability': 0.7101535535871051,
              'start': 32,
              'text': '范'}],
      '地点': [{'end': 66,
              'probability': 0.6405238209206026,
              'start': 54,
              'text': '本区卫清东路XXX号门口'}],
      '时间': [{'end': 10,
              'probability': 0.9186600131619826,
              'start': 0,
              'text': '2016年6月初某日'}]}]
    ```
    可见，在更加细分的涉毒类罪名法律文书的场景下，使用Taskflow的效果更不好。

### 2.训练定制
对于简单的抽取目标可以直接使用```paddlenlp.Taskflow```实现零样本（zero-shot）抽取，对于细分场景推荐使用轻定制功能（标注少量数据进行模型微调）以进一步提升效果。
#### 2.1 代码结构

```shell
.
├── utils.py          # 数据处理工具
├── model.py          # 模型组网脚本
├── doccano.py        # 数据标注脚本
├── doccano.md        # 数据标注文档
├── finetune.py       # 模型微调脚本
├── evaluate.py       # 模型评估脚本
└── README.md
```
#### 2.2 标注数据
#### 判决书场景
- 对于判决书数据，我们需要自己标注，生产数据集。

    使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，doccano导出数据后可通过[doccano.py](./uie/doccano.py)脚本将数据转换为输入模型时需要的形式。
    
    步骤详见 [doccano数据标注指南](./uie/doccano.md)，以及 [PaddleNLP开放域信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/3914778?channelType=0&channel=0)
    
    - STEP1 安装doccano
    
    ```shell
    pip install doccano
    ```
    - STEP2 初始化数据库与账户
    ```shell
    doccano init
    doccano createuser --username admin --password pass
    ```
    
    执行`doccano init`可能会出现以下报错：
    
    ```text
    sqlite3.OperationalError: no such function: JSON_VALID
    ```
    
    这是因为Python3.9之前的版本是默认不包含SQLLite JSON扩展。需要更新sqlite DLL才可以解决问题。解决方案: [sqlite3.OperationalError: no such function: JSON_VALID](http://www.chenxm.cc/article/1349.html.html)

    ```text
    django.db.utils.OperationalError: error in index django_celery_results_taskresult_hidden_cd77412f after drop column: no such column: hidden
    ```
    此时的解决方案为，更换Django版本为4.0.4：
    ```shell
    pip uninstall Django
    pip install Django==4.0.4
    ```
    当提示以下信息时，表示初始化成功：
    ```text
    Role created successfully "project_admin"
    Role created successfully "annotator"
    Role created successfully "annotation_approver"
    ```
    - STEP3 启动doccano
    
    首先，在终端中运行下面的代码来启动WebServer
    
    ```shell
    doccano webserver --port 8000
    ```
    然后，打开另一个终端，运行下面的代码启动任务队列：
    
    ```shell
    doccano task
    ```
    此时，我们就完成了doccano的启动。
    
    - STEP4 运行doccano来标注实体与关系
    
    从[上海市高级人民法院网](http://www.hshfy.sh.cn/shfy/gweb2017/flws_list_new.jsp)下载部分判决书，并转换成txt格式，一行即是一个标注子任务：[dataset.txt](./dataset.txt)
    ```text
    上海市宝山区人民法院 民事判决书 案号：（2022）沪0113民初6004号 原告：上海宝月物业管理有限公司，住所地上海市宝山区申浦路67号。 法定代表人：张建明，总经理。 委托诉讼代理人：陈永毅，该公司员工。 委托诉讼代理人：卢鑫，该公司员工。 被告：张洋，男，1983年6月24日出生，汉族，住上海市宝山区。 被告：马环环，女，1985年5月21日出生，汉族，住上海市宝山区。 委托诉讼代理人：张洋（系被告丈夫），住上海市宝山区。
    上海市第二中级人民法院 民事判决书 案号：（2022）沪02民终5444号 上诉人（原审被告）：上海微餐文化传播有限公司，住所地中国（上海）自由贸易试验区芳春路400号1幢3层。 法定代表人：郑虎勇，执行董事。 委托诉讼代理人：颜艳，上海九州通和（杭州）律师事务所律师。 被上诉人（原审原告）：谢斌澜，男，1970年8月25日出生，汉族，户籍所在地上海市静安区万荣路970弄32号102室。 委托诉讼代理人：贺小丽，北京市京师（上海）律师事务所律师。 委托诉讼代理人：汪鹏，北京市京师（上海）律师事务所律师。
    上海市宝山区人民法院 民事判决书 案号：（2022）沪0113民初7259号 原告：张祥林，男，1953年1月15日出生，汉族，住上海市嘉定区。 委托诉讼代理人：张文杰（系原告之子），男，1982年12月31日出生，汉族，住上海市嘉定区。 被告：任现令，男，1987年8月9日出生，汉族，住上海市宝山区。
    上海市高级人民法院 民事判决书 案号：（2021）沪民终1294号 上诉人（一审被告）：上海贡盈国际货运代理有限公司。 法定代表人：周吉蒙。 委托诉讼代理人：喻海，上海深度律师事务所律师。 委托诉讼代理人：朱泽荣，上海深度律师事务所律师。 被上诉人（一审原告）：池州豪迈机电贸易有限公司。 法定代表人：桂双龙。 委托诉讼代理人：许丽君。 委托诉讼代理人：杨维江，上海融孚律师事务所律师。 一审被告：上海华由船舶配件有限公司。 法定代表人：陶晖。 委托诉讼代理人：李晓清，远闻（上海）律师事务所律师。 委托诉讼代理人：范文菁，远闻（上海）律师事务所律师。 一审被告：上海越苒货物运输代理有限公司。 法定代表人：赵小刚。 一审被告：赵小刚。
    上海市徐汇区人民法院 民事判决书 案号：（2022）沪0104民初7935号 原告：咔么（上海）文化传播有限公司，住所地上海市宝山区友谊路1518弄10号1层H-188室。 法定代表人：李长伟，职务执行董事。 委托诉讼代理人：邓彬，上海市锦天城律师事务所律师。 委托诉讼代理人：何玉梅，上海市锦天城律师事务所律师。 被告：郭尘帆，男，2001年9月8日出生，汉族，住四川省遂宁市船山区。
    上海市徐汇区人民法院 民事判决书 案号：（2022）沪0104民初7926号 原告：张晓，男，1991年7月8日出生，汉族，住江苏省睢宁县。 被告：郝玉锦，女，1959年11月15日出生，汉族，住上海市徐汇区。 委托诉讼代理人：郑伟健（系郝玉锦配偶），男，住山西省阳泉市城区。
    上海市徐汇区人民法院 民事判决书 案号：（2022）沪0104民初7920号 原告：朱慧，女，1990年6月29日出生，汉族，住上海市徐汇区。 被告：戴金，男，1989年10月7日出生，汉族，住上海市浦东新区。
    上海金融法院 民事判决书 案号：（2022）沪74民终493号 上诉人（原审原告）：深圳市优信鹏达二手车经纪有限公司嘉兴第一分公司，营业场所浙江省嘉兴市南湖区文贤路1851号汽车商贸园二手车市场2号楼303室。 负责人：王桐。 委托诉讼代理人：徐凯佩，上海易锦律师事务所律师。 被上诉人（原审被告）：中国人寿财产保险股份有限公司嘉兴中心支公司，营业场所浙江省嘉兴市经济技术开发区秦逸路14号、18号、22号，华隆广场1幢201-202室、204室、301-305室、401-404室、407-410室、501-510室。 负责人：黄渊，总经理。 委托诉讼代理人：吕栋晓，男，公司员工。
    上海金融法院 民事判决书 案号：（2022）沪74民终450号 上诉人（原审被告）：中国平安财产保险股份有限公司上海分公司，营业场所上海市常熟路8号。 负责人：陈雪松，总经理。 委托诉讼代理人：周靖鲁，上海渲双律师事务所律师。 委托诉讼代理人：李毅臣，上海渲双律师事务所律师。 被上诉人（原审原告）：上海焜格金属制品厂，住所地上海市奉贤区沿钱公路5599号11080室。 负责人：朱小弟，经理。 委托诉讼代理人：方惠荣，上海达尊律师事务所律师。
    上海市第二中级人民法院 行政判决书 案号：（2022）沪02行终38号 上诉人（原审原告）唐建忠，男，1958年10月30日出生，汉族，住上海市浦东新区。 上诉人（原审原告）唐杰杰，男，1997年12月25日出生，汉族，住上海市浦东新区。 上诉人（原审原告）陈硕，男，1985年7月22日出生，汉族，住上海市浦东新区。 上诉人（原审原告）唐建国，男，1957年5月24日出生，汉族，住上海市浦东新区。 上诉人（原审原告）唐勇，男，1986年1月30日出生，汉族，住上海市浦东新区。 上诉人（原审第三人）唐建平，男，1965年11月23日出生，汉族，住上海市浦东新区。 上诉人（原审第三人）唐佩芳，女，1963年12月28日出生，汉族，住上海市浦东新区。 上诉人（原审第三人）张大满，女，1971年9月5日出生，汉族，住上海市浦东新区。 上述八名上诉人的共同委托代理人吴雯麒，上海旭灿律师事务所律师。 被上诉人（原审被告）上海市浦东新区北蔡镇人民政府，住所地上海市浦东新区沪南路1000号。 法定代表人徐建军，上海市浦东新区北蔡镇人民政府镇长。 出庭负责人黄祺炜，上海市浦东新区北蔡镇人民政府副镇长。 委托代理人顾娇娇，女，上海市浦东新区北蔡镇人民政府工作人员。 委托代理人郑方优，上海华宏律师事务所律师。 原审第三人唐秀娟，女，1954年3月10日出生，汉族，住上海市浦东新区。 原审第三人唐菊芳，女，1960年11月2日出生，汉族，住上海市浦东新区。
    上海金融法院 民事判决书 案号：（2022）沪74民终396号 上诉人（原审被告）：陈军，男，1973年11月18日出生，汉族，住江苏省连云港市海州区。 被上诉人（原审原告）：上海唯渡网络科技有限公司，住所地上海市长宁区镇宁路465弄161号4号楼116室。 法定代表人：武文秀，总经理。 委托诉讼代理人：王颖，上海瀛泰律师事务所律师。 委托诉讼代理人：段晨澜，上海瀛泰律师事务所律师。 原审被告：王东，男，1971年11月16日出生，汉族，住江苏省灌南县。 原审被告：李全浦，男，1966年8月8日出生，汉族，住江苏省连云港市赣榆区。
    上海市第二中级人民法院 民事判决书 案号：（2022）沪02民终4482号 上诉人（原审被告）：上海市警华农业信息科技有限公司，住所地上海市青浦区徐泾镇振泾路198号1幢2层A区175室。 法定代表人：徐金龙，总经理。 被上诉人（原审原告）：上海宏牛建设发展有限公司，住所地上海市青浦区公园路99号舜浦大厦7层A区748室。 法定代表人：田印芬，执行董事兼总经理。 委托诉讼代理人：孙慧凌，北京观韬中茂（上海）律师事务所律师。 委托诉讼代理人：周楠，北京观韬中茂（上海）律师事务所律师。 原审第三人：上海天时印刷有限公司，住所地上海市青浦区沪青平公路4501弄2号。 法定代表人：张小玲，董事长。 委托诉讼代理人：蒋西旺，北京惠诚律师事务所上海分所律师。 原审第三人：上海市青浦区人民政府夏阳街道办事处，住所地上海市青浦区外青松公路6300号。 法定代表人：徐川，主任。
    上海市第二中级人民法院 民事判决书 案号：（2022）沪02民终4311号 上诉人（原审原告）：张伯毅，男，1949年3月12日出生，汉族，户籍所在地上海市。 上诉人（原审原告）：金立彩，女，1952年12月5日出生，汉族，户籍所在地上海市。 上诉人（原审原告）：张某1，男，1982年9月23日出生，汉族，户籍所在地上海市。 上诉人（原审原告）：XXX，女，1982年8月22日出生，汉族，户籍所在地上海市。 上诉人（原审原告）：张某2，男，2008年9月25日出生，汉族，户籍所在地上海市。 法定代理人：张某1（系张某2之父），男，户籍所在地上海市。 法定代理人：XXX（系张某2之母），女，户籍所在地上海市。 上列五上诉人共同委托诉讼代理人：孙丹毅，上海汉盛律师事务所律师。 上诉人（原审被告）：张志毅，男，1956年4月16日出生，汉族，户籍所在地上海市。 委托诉讼代理人：栾伟强，上海劲力律师事务所律师。 委托诉讼代理人：高圳南，上海劲力律师事务所律师。
    上海市第二中级人民法院 民事判决书 案号：（2022）沪02民终4299号 上诉人（原审原告）：李荣芳，女，1957年7月18日出生，汉族，户籍所在地上海市。 委托诉讼代理人：蒋新中，上海正贯长虹律师事务所律师。 被上诉人（原审被告）：厉国义，男，1958年1月25日出生，汉族，户籍所在地上海市。 被上诉人（原审被告）：厉永智，男，1991年3月18日出生，汉族，户籍所在地上海市。 上列两被上诉人共同委托诉讼代理人：李杨，上海市天一律师事务所律师。
    ```
    导入docanno并完成实体标注与关系标注，再导出数据集，得到[admin.jsonl](./admin.jsonl)

#### 涉毒类法律文书场景

- 数据集内容见[./data/CAIL2022_ie](./data/CAIL2022_ie)

#### 2.3 抽取式任务数据转换
#### 判决书场景
  - 当标注完成后，在 doccano 平台上导出 `JSONL(relation)` 形式的文件，并将其重命名为 `doccano_ext.json` 后，放入 `./data` 目录下。
  - 通过 [doccano.py](./uie/doccano.py) 脚本进行数据形式转换，然后便可以开始进行相应模型训练。
  - 执行后会在[./data](./data)目录下生成训练/验证/测试集文件。
    ```shell
    python ./uie/doccano.py 
        --doccano_file ./data/doccano_ext.json 
        --task_type "ext" 
        --save_dir ./data 
        --negative_ratio 5
    ```
    
#### 涉毒类法律文书场景
  - 由于该数据集格式参考NYT数据集格式，PaddleNLP提供的数据形式转换工具无法直接使用，所以需要对其进行改造。详细改造内容可见[utils_cust.py](./uie/utils_cust.py)
  - 接着需要将[train.json](./data/CAIL2022_ie/train.json)放入`./data`目录下，并修改其实体名与关系名为中文。
  - todo 数据形式转换工具的改造还存在问题
  ```shell
    python ./uie/doccano_cust.py 
          --doccano_file ./data/small_train.json 
          --task_type "ext" 
          --save_dir ./data/drug 
          --negative_ratio 5
  ```
  - 之后会在[./data/drug](./data/drug)目录下生成训练/验证/测试集文件。
#### 2.4 模型微调
#### 判决书场景
tips: 推荐使用GPU环境，否则可能会内存溢出。CPU环境下，可以修改model为uie-tiny，适当调下batch_size。

[Linux下的PIP安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/linux-pip.html)

  使用3张GeForce 2080Ti进行多卡训练。
  ```shell
  python -u -m paddle.distributed.launch --gpus "1,2,7" ./finetune.py 
    --train_path ./data/train.txt 
    --dev_path ./data/dev.txt 
    --save_dir ./legal_judgment_checkpoint 
    --learning_rate 1e-5 
    --batch_size 8 
    --max_seq_len 512 
    --num_epochs 50 
    --model uie-base 
    --seed 1000 
    --logging_steps 10 
    --valid_steps 10
    --device gpu
  ```
可配置参数说明：

- `train_path`: 训练集文件路径。
- `dev_path`: 验证集文件路径。
- `save_dir`: 模型存储路径，默认为`./checkpoint`。
- `learning_rate`: 学习率，默认为1e-5。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `num_epochs`: 训练轮数，默认为100。
- `model`: 选择模型，程序会基于选择的模型进行模型微调，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `seed`: 随机种子，默认为1000.
- `logging_steps`: 日志打印的间隔steps数，默认10。
- `valid_steps`: evaluate的间隔steps数，默认100。
- `device`: 选用什么设备进行训练，可选cpu或gpu。

#### 涉毒类法律文书场景

对于涉毒类法律文书，以下是使用移动端GPU进行训练的过程:

注意：只需要下载CUDA、cuDNN、VisualStudio即可，不需要安装Anaconda。

[win10 安装Paddlepaddle-GPU](https://aistudio.baidu.com/aistudio/projectdetail/3383520?channelType=0&channel=0)

[官网教程--Windows下的PIP安装](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/windows-pip.html)

  物理机参数
  - 系统：Microsoft Windows [版本 10.0.19044.1826]
    - CPU：Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz   2.59 GHz
    - 内存：16.0 GB DDR4
    - GPU：NVIDIA GeForce RTX 3060 Laptop GPU
  
  如果使用安培架构显卡，即30系列显卡，必须安装11以上CUDA。30系显卡推荐安装11.2。

  如果您的计算机有NVIDIA® GPU，请确保满足以下条件并且安装GPU版PaddlePaddle
  
  CUDA 工具包10.1/10.2 配合 cuDNN 7 (cuDNN版本>=7.6.5）
  
  CUDA 工具包11.0配合cuDNN v8.0.4
  
  CUDA 工具包11.1配合cuDNN v8.1.1
  
  CUDA 工具包11.2配合cuDNN v8.1.1

  CUDA下载地址：[CUDA-toolkit-archive](https://developer.nvidia.cn/cuda-toolkit-archive)

  cuDNN下载地址：[cuDNN-toolkit-archive](https://developer.nvidia.cn/rdp/cudnn-archive)
  
  Microsoft VisualStudio 2019 Community下载地址：[Visual Studio 2019 版本 16.11](https://docs.microsoft.com/zh-cn/visualstudio/releases/2019/release-notes)

  ```shell
  nvidia-smi
  ```

  执行以上指令后发现本机安装的NVIDIA驱动版本对应的CUDA Driver版本为11.6，故下载CUDA工具包11.6配合cuDNN v8.4，paddlepaddle版本也选择CUDA11.6的版本。
  
  ```shell
  python -m pip install paddlepaddle-gpu==2.3.1.post116 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
  ```

单卡启动：
  ```shell
  python ./finetune.py 
      --train_path ../data/drug/train.txt 
      --dev_path ../data/drug/dev.txt 
      --save_dir ../checkpoint 
      --learning_rate 1e-5 
      --batch_size 4 
      --max_seq_len 512 
      --num_epochs 50 
      --model uie-base 
      --seed 1000 
      --logging_steps 10 
      --valid_steps 10 
      --device gpu
  ```
  当第一个epoch结束时，可能遇到如下报错：
  ```text
  Process finished with exit code -1073741819 (0xC0000005)
  ```
  解决方法是将Scipy版本降低至1.3.1
  ```shell
  pip install Scipy==1.3.1
  ```

#### 2.5 模型评估
通过运行以下命令进行模型评估：

```shell
python ./uie/evaluate.py 
    --model_path ./checkpoint/model_best 
    --test_path ./data/dev.txt 
    --batch_size 16 
    --max_seq_len 512
```

评估方式说明：采用单阶段评价的方式，即关系抽取、事件抽取等需要分阶段预测的任务对每一阶段的预测结果进行分别评价。验证/测试集默认会利用同一层级的所有标签来构造出全部负例。

可开启`debug`模式对每个正例类别分别进行评估，该模式仅用于模型调试：

```shell
python ./uie/evaluate.py 
    --model_path ./legal_judgment_checkpoint/model_best 
    --test_path ./data/dev.txt 
    --debug
```

输出打印示例：

```text
[2022-08-12 17:22:45,981] [    INFO] - -----------------------------
[2022-08-12 17:22:45,981] [    INFO] - Class Name: 原告
[2022-08-12 17:22:45,981] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-08-12 17:22:46,012] [    INFO] - -----------------------------
[2022-08-12 17:22:46,012] [    INFO] - Class Name: 法院
[2022-08-12 17:22:46,012] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-08-12 17:22:46,043] [    INFO] - -----------------------------
[2022-08-12 17:22:46,043] [    INFO] - Class Name: 被告
[2022-08-12 17:22:46,043] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-08-12 17:22:46,073] [    INFO] - -----------------------------
[2022-08-12 17:22:46,073] [    INFO] - Class Name: 原告委托代理人
[2022-08-12 17:22:46,073] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-08-12 17:22:46,103] [    INFO] - -----------------------------
[2022-08-12 17:22:46,103] [    INFO] - Class Name: 被告委托代理人
[2022-08-12 17:22:46,103] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
```

可配置参数说明：

- `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`及配置文件`model_config.json`。
- `test_path`: 进行评估的测试集文件。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `model`: 选择所使用的模型，可选有`uie-base`, `uie-medium`, `uie-mini`, `uie-micro`和`uie-nano`，默认为`uie-base`。
- `debug`: 是否开启debug模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。

#### 2.6 定制模型预测

`paddlenlp.Taskflow`装载定制模型，通过`task_path`指定模型权重文件的路径，路径下需要包含训练好的模型权重文件`model_state.pdparams`。

- 对于判决书场景：
    ```python
    >>> schema = ['法院', {'原告': '原告委托代理人'}, {'被告': '被告委托代理人'}]  # Define the schema for entity extraction
    >>> ie = Taskflow('information_extraction', schema=schema, task_path='./checkpoint/model_best') # task_path='./checkpoint/model_best'
    >>> pprint(ie("上海金融法院"
              "民事判决书"
              "案号：（2022）沪74民终186号"
              "上诉人（原审被告）：潢川县发展投资有限责任公司，住所地河南省信阳市潢川县财政局。"
              "法定代表人：梅利平，董事长。"
              "委托诉讼代理人：李明修，该公司员工。"
              "委托诉讼代理人：李阳，河南捷达律师事务所律师。"
              "被上诉人（原审原告）：海发宝诚融资租赁有限公司（原名中远海运租赁有限公司），住所地中国（上海）自由贸易试验区福山路450号3E室。"
              "法定代表人：陈易明，总经理。"
              "委托诉讼代理人：邹华恩，北京德和衡（上海）律师事务所律师。"
              "委托诉讼代理人：王坤，北京德和衡（上海）律师事务所律师。"))
  [{'原告': [{'end': 147,
            'probability': 0.9939836124423422,
            'relations': {'原告委托代理人': [{'end': 214,
                                       'probability': 0.9940615194536804,
                                       'start': 211,
                                       'text': '邹华恩'},
                                      {'end': 242,
                                       'probability': 0.9902140498778493,
                                       'start': 240,
                                       'text': '王坤'}]},
            'start': 135,
            'text': '海发宝诚融资租赁有限公司'}],
    '法院': [{'end': 6,
            'probability': 0.9994039150743141,
            'start': 0,
            'text': '上海金融法院'}],
    '被告': [{'end': 52,
            'probability': 0.9978744305253144,
            'relations': {'被告委托代理人': [{'end': 111,
                                       'probability': 0.9736823939226369,
                                       'start': 109,
                                       'text': '李阳'},
                                      {'end': 94,
                                       'probability': 0.9666654534422037,
                                       'start': 91,
                                       'text': '李明修'}]},
            'start': 39,
            'text': '潢川县发展投资有限责任公司'}]}]
    ```
  从预测结果来看，训练出的模型能够正确标注出schema里的实体与关系。
- 对于涉毒类法律文书场景
  ```python
  >>> schema = ['地点', '时间', '毒品重量', {'涉案人': '涉案人'}, {'涉案人': '毒品类型'}]  
  >>> ie = Taskflow('information_extraction', schema=schema, task_path='./drug_checkpoint/model_best')
  >>> pprint(ie("2016年6月初某日，被告人陆xx经与吸毒人员范xx事先联系，由范xx驾驶牌号为沪C8XXXX红色奥迪轿车至本区卫清东路XXX号门口，被告人陆xx进入该车内将约0.3克甲基苯丙胺以人民币300元的价格贩卖给范xx"))
  [{'地点': [{'end': 66,
          'probability': 0.9889165838872458,
          'start': 56,
          'text': '卫清东路XXX号门口'}],
  '时间': [{'end': 10,
          'probability': 0.9955922727144042,
          'start': 0,
          'text': '2016年6月初某日'}],
  '毒品重量': [{'end': 84,
            'probability': 0.955537631782633,
            'start': 80,
            'text': '0.3克'}],
  '涉案人': [{'end': 106,
           'probability': 0.5274689057934552,
           'relations': {'毒品类型': [{'end': 89,
                                   'probability': 0.9990948659401155,
                                   'start': 84,
                                   'text': '甲基苯丙胺'}]},
           'start': 103,
           'text': '范xx'},
          {'end': 17,
           'probability': 0.8662895950264513,
           'relations': {'毒品类型': [{'end': 89,
                                   'probability': 0.9991567675652391,
                                   'start': 84,
                                   'text': '甲基苯丙胺'}]},
           'start': 14,
           'text': '陆xx'},
          {'end': 26,
           'probability': 0.7007867062752524,
           'relations': {'毒品类型': [{'end': 89,
                                   'probability': 0.9990948659401155,
                                   'start': 84,
                                   'text': '甲基苯丙胺'}]},
           'start': 23,
           'text': '范xx'},
          {'end': 73,
           'probability': 0.7100993406556988,
           'relations': {'毒品类型': [{'end': 89,
                                   'probability': 0.9991567675652391,
                                   'start': 84,
                                   'text': '甲基苯丙胺'}]},
           'start': 70,
           'text': '陆xx'},
          {'end': 106,
           'probability': 0.5274689057934552,
           'start': 103,
           'text': '范xx'},
          {'end': 17,
           'probability': 0.8662895950264513,
           'start': 14,
           'text': '陆xx'},
          {'end': 26,
           'probability': 0.7007867062752524,
           'start': 23,
           'text': '范xx'},
          {'end': 73,
           'probability': 0.7100993406556988,
           'start': 70,
           'text': '陆xx'}]}]
  ```
  标注结果相比zero-shot正确率有所提升。




