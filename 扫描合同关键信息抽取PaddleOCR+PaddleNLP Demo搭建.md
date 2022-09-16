# 扫描合同关键信息抽取PaddleOCR+PaddleNLP Demo搭建
1. 使用PaddleOCR提取扫描文本内容
2. 使用PaddleNLP抽取自定义信息

![image](https://user-images.githubusercontent.com/61258341/190606142-8258ae53-9ba6-40b3-82d8-1e21c05dfb44.png)


# 0. 环境搭建

使用满足CUDA≥11.2的Docker容器进行部署

1）制作CUDA≥11.2的Python部署镜像

requirements.txt文件内容：

```
paddlenlp==2.3.5
paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
onnx
onnxconverter_common
onnxruntime-gpu
minio
tqdm
gunicorn
flask
paddleocr
```

Dockerfile文件内容：

```docker
FROM paddlepaddle/paddle:2.3.2-gpu-cuda11.2-cudnn8
WORKDIR /opt/paddleocr
COPY requirements.txt requirements.txt
# 安装依赖
RUN pip config set global.index-url https://pypi.douban.com/simple
RUN pip install --no-cache-dir -r requirements.txt
```

2）构建镜像

`podman build -t 2.3.2-gpu-cuda11.2-cudnn8:paddleocr .`

3）启动容器

启动容器：

`podman run --name ocr-nlp --shm-size=2g -it -v $PWD:/opt/paddleocr 2.3.2-gpu-cuda11.2-cudnn8:paddleocr /bin/bash`

安装依赖：

`python -m pip install opencv-contrib-python==4.4.0.46`

4）当您需要第二次进入Docker容器中，使用如下命令：

- 启动之前创建的容器

`podman start ocr-nlp`

- 进入启动的容器

`podman attach ocr-nlp`

# 1. 扫描合同文本内容提取

## 1.1 效果测试

使用一张合同图片作为测试样本，感受ppocrv3模型效果：

![image](https://user-images.githubusercontent.com/61258341/190606254-fc632612-ac9e-4c7f-b414-6fffc8f050d1.png)

使用中文检测+识别模型提取文本，实例化PaddleOCR类：

```python
from paddleocr import PaddleOCR, draw_ocr

# paddleocr目前支持中英文、英文、法语、德语、韩语、日语等80个语种，可以通过修改lang参数进行切换
ocr = PaddleOCR(use_angle_cls=False, lang="ch")  # need to run only once to download and load model into memory
```

一行命令启动预测，预测结果包括`检测框`和`文本识别内容`:

```python
img_path = "./test_img/hetong2.jpg"
result = ocr.ocr(img_path, cls=False)
for line in result:
    print(line)
```

```
[2022/09/16 07:20:23] ppocr DEBUG: dt_boxes num : 21, elapse : 1.3216724395751953
[2022/09/16 07:20:23] ppocr DEBUG: rec_res num  : 21, elapse : 0.6431457996368408
[[[1119.0, 329.0], [1388.0, 329.0], [1388.0, 424.0], [1119.0, 424.0]], ('合同书', 0.9876987934112549)]
[[[381.0, 573.0], [1045.0, 581.0], [1044.0, 669.0], [380.0, 661.0]], ('甲方：叶县境保护局', 0.9824250340461731)]
[[[388.0, 723.0], [1370.0, 734.0], [1369.0, 800.0], [387.0, 789.0]], ('乙方：叶县建昆市政工程有限公司', 0.9463322162628174)]
[[[514.0, 851.0], [2105.0, 866.0], [2104.0, 939.0], [513.0, 924.0]], ('依照《中华人民共和国合同法》、《中华人民共和国建筑', 0.9392094612121582)]
[[[381.0, 982.0], [2094.0, 1001.0], [2093.0, 1067.0], [380.0, 1048.0]], ('法》及其他有关法规，经公开招投标，依据中标通知书及相', 0.9737498164176941)]
[[[384.0, 1114.0], [2097.0, 1133.0], [2097.0, 1195.0], [384.0, 1176.0]], ('关材料，甲乙双方充分友好协商，就叶县环境空气质量自动', 0.9139176607131958)]
[[[381.0, 1238.0], [1595.0, 1250.0], [1595.0, 1323.0], [380.0, 1311.0]], ('监测站建设工程达成一致，特订立本合同。', 0.9688721895217896)]
[[[539.0, 1381.0], [923.0, 1381.0], [923.0, 1443.0], [539.0, 1443.0]], ('一、工程概况', 0.8555393218994141)]
[[[510.0, 1501.0], [1964.0, 1520.0], [1964.0, 1582.0], [509.0, 1563.0]], ('工程名称：叶县环境空气质量自动监测站建设工程', 0.9296047687530518)]
[[[506.0, 1633.0], [971.0, 1633.0], [971.0, 1706.0], [506.0, 1706.0]], ('工程地点：叶县', 0.9307069778442383)]
[[[506.0, 1761.0], [1894.0, 1779.0], [1893.0, 1845.0], [505.0, 1826.0]], ('工程内容：施工图设计范围内招标人发包的内容', 0.9445862770080566)]
[[[503.0, 1892.0], [1292.0, 1900.0], [1292.0, 1966.0], [502.0, 1958.0]], ('承包范围：详见工程量清单', 0.976337194442749)]
[[[502.0, 2024.0], [1226.0, 2028.0], [1226.0, 2093.0], [502.0, 2089.0]], ('二：工程质量标准：合格', 0.9450864195823669)]
[[[503.0, 2148.0], [2090.0, 2167.0], [2089.0, 2240.0], [502.0, 2221.0]], ('三：合同价格：总价为人民币大写：参拾玖万捌仟伍佰', 0.9563236236572266)]
[[[366.0, 2276.0], [2090.0, 2298.0], [2089.0, 2375.0], [365.0, 2352.0]], ('元，小写：398500.00元。总价中包括站房工程建设、安装', 0.9760955572128296)]
[[[370.0, 2411.0], [2079.0, 2433.0], [2078.0, 2499.0], [369.0, 2477.0]], ('及相关避雷、消防、接地、电力、材料费、检验费、安全、', 0.9455706477165222)]
[[[366.0, 2542.0], [1510.0, 2557.0], [1509.0, 2620.0], [365.0, 2604.0]], ('验收等所需费用及其他相关费用和税金。', 0.9580625891685486)]
[[[499.0, 2674.0], [2042.0, 2696.0], [2041.0, 2758.0], [498.0, 2736.0]], ('四、乙方保证站房建设需符合环保部HJ655-2013，、', 0.9152480363845825)]
[[[359.0, 2798.0], [2079.0, 2828.0], [2078.0, 2890.0], [358.0, 2860.0]], ('HJ193-2013技术规范要求和国家相关规定。材料符合国家消', 0.9377042651176453)]
[[[359.0, 2929.0], [2075.0, 2959.0], [2074.0, 3022.0], [358.0, 2992.0]], ('防B+1级要求。该房建好后做的三不透，不透光、不透雨、', 0.9151259660720825)]
[[[359.0, 3061.0], [2072.0, 3091.0], [2070.0, 3153.0], [358.0, 3123.0]], ('不透风。在楼顶安装安全设施、避雷设施和仪器前端加装防', 0.9607052206993103)]
```

结果可视化：

```python
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.show()
im_show.save("./test_img/ocr_hetong2.jpg")
```

![image](https://user-images.githubusercontent.com/61258341/190606321-56ea9033-5b6b-42f1-8f66-ce74ad839256.png)

## 1.2 ****图片预处理****

通过上图可视化结果可以看到，印章部分造成的文本遮盖，影响了文本识别结果，因此可以考虑通道提取，去除图片中的红色印章：

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读入图像,三通道
image=cv2.imread("./test_img/hetong2.jpg",cv2.IMREAD_COLOR) #timg.jpeg

#获得三个通道
Bch,Gch,Rch=cv2.split(image) 

#保存三通道图片
cv2.imwrite('blue_channel.jpg',Bch)
cv2.imwrite('green_channel.jpg',Gch)
cv2.imwrite('red_channel.jpg',Rch)
```

## 1.3 合同文本信息提取

经过图片预处理后，合同照片的红色通道被分离，获得了一张相对更干净的图片，此时可以再次使用ppocr模型提取文本内容：

```python
ocr = PaddleOCR(use_angle_cls=False, lang="ch")  # need to run only once to download and load model into memory
img_path = './red_channel.jpg'
result = ocr.ocr(img_path, cls=False)

# 可视化结果
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./simfang.ttf')
im_show = Image.fromarray(im_show)
vis = np.array(im_show)
im_show.show()
im_show.save("./ocr_red_channel.jpg")
```

忽略检测框内容，提取完整的合同文本，并保存为txt文件，方便后续进行信息抽取：

```python
all_context = "\n".join(txts)
print(all_context)
with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(all_context)
```

```
合同书
甲方：叶县环境保护局
乙方：叶县建昆市政工程有限公司
依照《中华人民共和国合同法》、《中华人民共和国建筑
法》及其他有关法规，经公开招投标，依据中标通知书及相
关材料，甲乙双方充分友好协商，就叶县环境空气质量自动
监测站建设工程达成一致，特订立本合同。
一、工程概况
工程名称：叶县环境空气质量自动监测站建设工程
工程地点：叶县
工程内容：施工图设计范围内招标人发包的内容
承包范围：详见工程量清单
二：工程质量标准：合格
三：合同价格：总价为人民币大写：参拾玖万捌仟伍佰
元，小写：398500.00元。总价中包括站房工程建设、安装
及相关避雷、消防、接地、电力、材料费、检验费、安全、
验收等所需费用及其他相关费用和税金。
四、乙方保证站房建设需符合环保部HJ655-2013，、
HJ193-2013技术规范要求和国家相关规定。材料符合国家消
防B+1级要求。该房建好后做的三不透，不透光、不透雨、
不透风。在楼顶安装安全设施、避雷设施和仪器前端加装防
```

通过以上环节就完成了扫描合同关键信息抽取的第一步：文本内容提取，接下来可以基于识别出的文本内容抽取关键信息。

## 1.3.4 合同关键信息抽取

将使用OCR提取好的文本作为输入，进行关键信息抽取：

```python
schema = ["甲方", "乙方", "总价"]
ie = Taskflow('information_extraction', schema=schema)
ie.set_schema(schema)

all_context = ''
with open('result.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        all_context += line

pprint(ie(all_context))
```

```json
[{'乙方': [{'end': 28,
          'probability': 0.8460316512985067,
          'start': 16,
          'text': '叶县建昆市政工程有限公司'}],
  '总价': [{'end': 228,
          'probability': 0.36525866253101924,
          'start': 219,
          'text': '参拾玖万捌仟伍佰元'}],
  '甲方': [{'end': 13,
          'probability': 0.9178748049301255,
          'start': 6,
          'text': '叶县环境保护局'}]}]
```

可以看到UIE模型可以准确的提取出关键信息。

# 2. 效果优化

实际图片采集过程中，可能出现部分图片弯曲等问题，导致使用默认参数识别文本时存在漏检，影响关键信息获取。

![image](https://user-images.githubusercontent.com/61258341/190606463-98de1a36-3636-457c-9bae-8aba87e9c1c5.png)



可视化结果可以看到，弯曲图片存在漏检，一般来说可以通过调整后处理参数解决，无需重新训练模型。漏检问题往往是因为检测模型获得的分割图太小，生成框的得分过低被过滤掉了，通常有两种方式调整参数：

- 开启`use_dilatiion=True` 膨胀分割区域
- 调小`det_db_box_thresh`阈值

```python
# 重新实例化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="ch", det_db_box_thresh=0.3, use_dilation=True)

# 预测并可视化
img_path = "./test_img/hetong3.jpg"
# 预测结果
result = ocr.ocr(img_path, cls=False)
# 可视化结果
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.show()
```

![image](https://user-images.githubusercontent.com/61258341/190606485-bcb154b4-82eb-4717-949b-920a26393828.png)

可以看到漏检问题被很好的解决。
