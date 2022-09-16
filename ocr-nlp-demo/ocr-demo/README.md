# 多模态表单识别PaddleOCR Demo搭建
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

从Git上拉取并进入PaddleOCR的目录：

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
```

启动容器：

`podman run --name paddle_docker --shm-size=2g -it -v $PWD:/opt/paddleocr 2.3.2-gpu-cuda11.2-cudnn8:paddleocr /bin/bash`

安装依赖环境：

`pip install -r requirements.txt`

`pip install -U https://paddleocr.bj.bcebos.com/ppstructure/whl/paddlenlp-2.3.0.dev0-py3-none-any.whl`

4）当您需要第二次进入Docker容器中，使用如下命令：

- 启动之前创建的容器

`docker start paddle_docker`

- 进入启动的容器

`docker attach paddle_docker`

# 1. 数据集下载

使用PaddleOCR事先处理好的XFUND_zh数据集进行快速体验。

```bash
mkdir train_data
cd train_data
wget [https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar](https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar) && tar -xf XFUND.tar
cd ..
```

# 2. 下载预训练模型

PaddleOCR提供了训练脚本、评估脚本和预测脚本，本节将以 VI-LayoutXLM 多模态预训练模型为例进行讲解。

使用下面的方法，下载基于XFUND数据的SER与RE任务预训练模型。

```bash
mkdir pretrained_model
cd pretrained_model
# 下载并解压SER预训练模型
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_pretrained.tar
tar -xf ser_vi_layoutxlm_xfund_pretrained.tar

# 下载并解压RE预训练模型
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/re_vi_layoutxlm_xfund_pretrained.tar
tar -xf re_vi_layoutxlm_xfund_pretrained.tar
```

# 3. 模型评估与预测

## 3.1 指标评估

训练中模型参数默认保存在`Global.save_model_dir`目录下。

在评估指标时，需要设置`Architecture.Backbone.checkpoints`指向保存的参数文件。若使用自训练的模型，则默认保存目录为: `./output/ser_vi_layoutxlm_xfund_zh/best_accuracy`，这里使用下载的预训练模型进行评估，所以设置路径为：`./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy`

评估数据集可以通过 `configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml` 修改Eval中的 `label_file_path` 设置。

```bash
# GPU 评估， Global.checkpoints 为待测权重
python3 tools/eval.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy
```

评估时可能会遇到如下报错：

```
[2022-09-16 02:15:27,889] [   ERROR] - Downloading from https://bj.bcebos.com/paddlenlp/models/community/vi-layoutxlm-base-uncased/model_state.pdparams failed with code 404!
......
RuntimeError: Can't load weights for 'vi-layoutxlm-base-uncased'.
......
```

此时需要更新paddlenlp包，运行如下指令：

```bash
pip install -U https://paddleocr.bj.bcebos.com/ppstructure/whl/paddlenlp-2.3.0.dev0-py3-none-any.whl
```

安装完成后，重新运行评估指令输出信息如下：

```
[2022/09/16 02:40:11] ppocr INFO: metric eval ***************
[2022/09/16 02:40:11] ppocr INFO: precision:0.9146722164412071
[2022/09/16 02:40:11] ppocr INFO: recall:0.9499279538904899
[2022/09/16 02:40:11] ppocr INFO: hmean:0.9319667785827884
[2022/09/16 02:40:11] ppocr INFO: fps:28.27982401704056
```

## 3.2 测试信息抽取结果

使用如下命令进行中文模型预测

```bash
python3 tools/infer_kie_token_ser.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy Global.infer_img=./ppstructure/docs/kie/input/zh_val_42.jpg
```

预测图片如下所示，图片会存储在`Global.save_res_path`路径中。

![zh_val_42_ser](https://user-images.githubusercontent.com/61258341/190610613-26730ff2-0ec5-440a-b6d6-1351c6fcffdb.jpg)

预测过程中，默认会加载PP-OCRv3的检测识别模型，用于OCR的信息抽取，如果希望加载预先获取的OCR结果，可以使用下面的方式进行预测，指定`Global.infer_img`为标注文件，其中包含图片路径以及OCR信息，同时指定`Global.infer_mode`为False，表示此时不使用OCR预测引擎。

```bash
python3 tools/infer_kie_token_ser.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy Global.infer_img=./train_data/XFUND/zh_val/val.json Global.infer_mode=False
```

对于上述图片，如果使用标注的OCR结果进行信息抽取，预测结果如下。

![zh_val_42_ser](https://user-images.githubusercontent.com/61258341/190610720-7d55f431-8ad0-4235-9ff9-097013043556.jpg)

可以看出，部分检测框信息更加准确，但是整体信息抽取识别结果基本一致。

# 4. 模型导出与预测

## 4.1 模型导出

信息抽取模型中的SER任务转静态参数模型步骤如下：

```bash
# -c 后面设置训练算法的yml配置文件
# -o 配置可选参数
# Architecture.Backbone.checkpoints 参数设置待转换的训练模型地址
# Global.save_inference_dir 参数设置转换的模型将保存的地址

python3 tools/export_model.py -c configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml -o Architecture.Backbone.checkpoints=./pretrained_model/ser_vi_layoutxlm_xfund_pretrained/best_accuracy Global.save_inference_dir=./inference/ser_vi_layoutxlm
```

转换成功后，在目录下有三个文件：

```
inference/ser_vi_layoutxlm/
    ├── inference.pdiparams         # inference模型的参数文件
    ├── inference.pdiparams.info    # inference模型的参数信息，可忽略
    └── inference.pdmodel           # inference模型的模型结构文件
```

## 4.2 模型推理

VI-LayoutXLM模型基于SER任务进行推理，可以执行如下命令：

```bash
cd ppstructure
python3 kie/predict_kie_token_ser.py \
  --kie_algorithm=LayoutXLM \
  --ser_model_dir=../inference/ser_vi_layoutxlm \
  --image_dir=./docs/kie/input/zh_val_42.jpg \
  --ser_dict_path=../train_data/XFUND/class_list_xfun.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx"
```

可视化SER结果结果默认保存到`./output`文件夹里面。结果示例如下：

![zh_val_42_ser](https://user-images.githubusercontent.com/61258341/190610814-b4212b84-9bd9-4393-b8c4-3b9ccfa43347.jpg)
