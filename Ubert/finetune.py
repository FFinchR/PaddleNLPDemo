import argparse
from pprint import pprint

from fengshen import UbertPiplines
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3,4,7'


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser = UbertPiplines.piplines_args(total_parser)
    args = total_parser.parse_args()

    # 设置一些训练要使用到的参数
    args.pretrained_model_path = 'IDEA-CCNL/Erlangshen-Ubert-110M-Chinese' #预训练模型的路径，我们提供的预训练模型存放在HuggingFace上
    args.default_root_dir = './'  #默认主路径，用来放日志、tensorboard等
    args.max_epochs = 1000
    args.check_val_every_n_epoch = 10
    args.gpus = 3
    args.batch_size = 1

    # 只需要将数据处理成为下面数据的 json 样式就可以一键训练和预测，下面只是提供了一条示例样本
    train_data = [
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市宝山区人民法院 民事判决书 案号：（2022）沪0113民初6004号 原告：上海宝月物业管理有限公司，住所地上海市宝山区申浦路67号。 法定代表人：张建明，总经理。 "
                    "委托诉讼代理人：陈永毅，该公司员工。 委托诉讼代理人：卢鑫，该公司员工。 被告：张洋，男，1983年6月24日出生，汉族，住上海市宝山区。 "
                    "被告：马环环，女，1985年5月21日出生，汉族，住上海市宝山区。 委托诉讼代理人：张洋（系被告丈夫），住上海市宝山区。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市宝山区人民法院", "entity_type": "法院", "entity_idx": [[0, 10]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "上海宝月物业管理有限公司", "entity_type": "原告", "entity_idx": [[42, 54]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "陈永毅", "entity_type": "原告委托代理人", "entity_idx": [[87, 94]]},
                    {"entity_name": "卢鑫", "entity_type": "原告委托代理人", "entity_idx": [[106, 113]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "张洋", "entity_type": "被告", "entity_idx": [[127, 129]]},
                    {"entity_name": "马环环", "entity_type": "被告", "entity_idx": [[160, 163]]},
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "张洋", "entity_type": "被告委托代理人", "entity_idx": [[199, 201]]}
                ]}
            ],
            "id": 0},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市第二中级人民法院 民事判决书 案号：（2022）沪02民终5444号 上诉人（原审被告）：上海微餐文化传播有限公司，住所地中国（上海）自由贸易试验区芳春路400号1幢3层。 "
                    "法定代表人：郑虎勇，执行董事。 委托诉讼代理人：颜艳，上海九州通和（杭州）律师事务所律师。 "
                    "被上诉人（原审原告）：谢斌澜，男，1970年8月25日出生，汉族，户籍所在地上海市静安区万荣路970弄32号102室。 委托诉讼代理人：贺小丽，北京市京师（上海）律师事务所律师。 "
                    "委托诉讼代理人：汪鹏，北京市京师（上海）律师事务所律师。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市第二中级人民法院", "entity_type": "法院", "entity_idx": [[0, 11]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "谢斌澜", "entity_type": "原告", "entity_idx": [[147, 150]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "贺小丽", "entity_type": "原告委托代理人", "entity_idx": [[204, 207]]},
                    {"entity_name": "汪鹏", "entity_type": "原告委托代理人", "entity_idx": [[234, 236]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "上海微餐文化传播有限公司", "entity_type": "被告", "entity_idx": [[48, 60]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "颜艳", "entity_type": "被告委托代理人", "entity_idx": [[114, 116]]}
                ]}
            ],
            "id": 1},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市宝山区人民法院 民事判决书 案号：（2022）沪0113民初7259号 原告：张祥林，男，1953年1月15日出生，汉族，住上海市嘉定区。 "
                    "委托诉讼代理人：张文杰（系原告之子），男，1982年12月31日出生，汉族，住上海市嘉定区。 被告：任现令，男，1987年8月9日出生，汉族，住上海市宝山区。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市宝山区人民法院", "entity_type": "法院", "entity_idx": [[0, 10]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "张祥林", "entity_type": "原告", "entity_idx": [[42, 45]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "张文杰", "entity_type": "原告委托代理人", "entity_idx": [[81, 84]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "任现令", "entity_type": "被告", "entity_idx": [[123, 126]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": []}
            ],
            "id": 2},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市高级人民法院 民事判决书 案号：（2021）沪民终1294号 上诉人（一审被告）：上海贡盈国际货运代理有限公司。 法定代表人：周吉蒙。 委托诉讼代理人：喻海，上海深度律师事务所律师。 "
                    "委托诉讼代理人：朱泽荣，上海深度律师事务所律师。 被上诉人（一审原告）：池州豪迈机电贸易有限公司。 法定代表人：桂双龙。 委托诉讼代理人：许丽君。 "
                    "委托诉讼代理人：杨维江，上海融孚律师事务所律师。 一审被告：上海华由船舶配件有限公司。 法定代表人：陶晖。 委托诉讼代理人：李晓清，远闻（上海）律师事务所律师。 "
                    "委托诉讼代理人：范文菁，远闻（上海）律师事务所律师。 一审被告：上海越苒货物运输代理有限公司。 法定代表人：赵小刚。 一审被告：赵小刚。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市高级人民法院", "entity_type": "法院", "entity_idx": [[0, 9]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "池州豪迈机电贸易有限公司", "entity_type": "原告", "entity_idx": [[131, 143]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "许丽君", "entity_type": "原告委托代理人", "entity_idx": [[164, 167]]},
                    {"entity_name": "杨维江", "entity_type": "原告委托代理人", "entity_idx": [[177, 180]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "上海贡盈国际货运代理有限公司", "entity_type": "被告", "entity_idx": [[44, 58]]},
                    {"entity_name": "上海华由船舶配件有限公司", "entity_type": "被告", "entity_idx": [[199, 211]]},
                    {"entity_name": "上海越苒货物运输代理有限公司", "entity_type": "被告", "entity_idx": [[282, 296]]},
                    {"entity_name": "赵小刚", "entity_type": "被告", "entity_idx": [[314, 317]]}

                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "喻海", "entity_type": "被告委托代理人", "entity_idx": [[79, 81]]},
                    {"entity_name": "朱泽荣", "entity_type": "被告委托代理人", "entity_idx": [[103, 106]]},
                    {"entity_name": "李晓清", "entity_type": "被告委托代理人", "entity_idx": [[231, 234]]},
                    {"entity_name": "范文菁", "entity_type": "被告委托代理人", "entity_idx": [[258, 261]]}

                ]}
            ],
            "id": 3},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市徐汇区人民法院 民事判决书 案号：（2022）沪0104民初7935号 原告：咔么（上海）文化传播有限公司，住所地上海市宝山区友谊路1518弄10号1层H-188室。 "
                    "法定代表人：李长伟，职务执行董事。 委托诉讼代理人：邓彬，上海市锦天城律师事务所律师。 委托诉讼代理人：何玉梅，上海市锦天城律师事务所律师。 "
                    "被告：郭尘帆，男，2001年9月8日出生，汉族，住四川省遂宁市船山区。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市徐汇区人民法院", "entity_type": "法院", "entity_idx": [[0, 10]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "咔么（上海）文化传播有限公司", "entity_type": "原告", "entity_idx": [[42, 56]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "邓彬", "entity_type": "原告委托代理人", "entity_idx": [[113, 115]]},
                    {"entity_name": "何玉梅", "entity_type": "原告委托代理人", "entity_idx": [[139, 142]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "郭尘帆", "entity_type": "被告", "entity_idx": [[161, 164]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": []}
            ],
            "id": 4},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市徐汇区人民法院 民事判决书 案号：（2022）沪0104民初7926号 原告：张晓，男，1991年7月8日出生，汉族，住江苏省睢宁县。 "
                    "被告：郝玉锦，女，1959年11月15日出生，汉族，住上海市徐汇区。 委托诉讼代理人：郑伟健（系郝玉锦配偶），男，住山西省阳泉市城区。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市徐汇区人民法院", "entity_type": "法院", "entity_idx": [[0, 10]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "张晓", "entity_type": "原告", "entity_idx": [[42, 44]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": []},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "郝玉锦", "entity_type": "被告", "entity_idx": [[74, 77]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "郑伟健", "entity_type": "被告委托代理人", "entity_idx": [[114, 117]]}
                ]}
            ],
            "id": 5},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市徐汇区人民法院 民事判决书 案号：（2022）沪0104民初7920号 原告：朱慧，女，1990年6月29日出生，汉族，住上海市徐汇区。 "
                    "被告：戴金，男，1989年10月7日出生，汉族，住上海市浦东新区。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市徐汇区人民法院", "entity_type": "法院", "entity_idx": [[0, 10]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "朱慧", "entity_type": "原告", "entity_idx": [[42, 44]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": []},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "戴金", "entity_type": "被告", "entity_idx": [[75, 77]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": []}
            ],
            "id": 6},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海金融法院 民事判决书 案号：（2022）沪74民终450号 上诉人（原审被告）：中国平安财产保险股份有限公司上海分公司，营业场所上海市常熟路8号。 负责人：陈雪松，总经理。 "
                    "委托诉讼代理人：周靖鲁，上海渲双律师事务所律师。 委托诉讼代理人：李毅臣，上海渲双律师事务所律师。 被上诉人（原审原告）：上海焜格金属制品厂，住所地上海市奉贤区沿钱公路5599号11080室。 "
                    "负责人：朱小弟，经理。 委托诉讼代理人：方惠荣，上海达尊律师事务所律师。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海金融法院", "entity_type": "法院", "entity_idx": [[0, 6]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "上海焜格金属制品厂", "entity_type": "原告",
                     "entity_idx": [[150, 159]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "方惠荣", "entity_type": "原告委托代理人", "entity_idx": [[206, 209]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "中国平安财产保险股份有限公司上海分公司", "entity_type": "被告",
                     "entity_idx": [[42, 61]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "周靖鲁", "entity_type": "被告委托代理人", "entity_idx": [[97, 100]]},
                    {"entity_name": "李毅臣", "entity_type": "被告委托代理人", "entity_idx": [[122, 125]]},
                ]}
            ],
            "id": 7},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市第二中级人民法院 行政判决书 案号：（2022）沪02行终38号 上诉人（原审原告）唐建忠，男，1958年10月30日出生，汉族，住上海市浦东新区。 "
                    "上诉人（原审原告）唐杰杰，男，1997年12月25日出生，汉族，住上海市浦东新区。 上诉人（原审原告）陈硕，男，1985年7月22日出生，汉族，住上海市浦东新区。 "
                    "上诉人（原审原告）唐建国，男，1957年5月24日出生，汉族，住上海市浦东新区。 上诉人（原审原告）唐勇，男，1986年1月30日出生，汉族，住上海市浦东新区。 "
                    "上诉人（原审第三人）唐建平，男，1965年11月23日出生，汉族，住上海市浦东新区。 上诉人（原审第三人）唐佩芳，女，1963年12月28日出生，汉族，住上海市浦东新区。 "
                    "上诉人（原审第三人）张大满，女，1971年9月5日出生，汉族，住上海市浦东新区。 上述八名上诉人的共同委托代理人吴雯麒，上海旭灿律师事务所律师。 "
                    "被上诉人（原审被告）上海市浦东新区北蔡镇人民政府，住所地上海市浦东新区沪南路1000号。 法定代表人徐建军，上海市浦东新区北蔡镇人民政府镇长。 "
                    "出庭负责人黄祺炜，上海市浦东新区北蔡镇人民政府副镇长。 委托代理人顾娇娇，女，上海市浦东新区北蔡镇人民政府工作人员。 委托代理人郑方优，上海华宏律师事务所律师。 "
                    "原审第三人唐秀娟，女，1954年3月10日出生，汉族，住上海市浦东新区。 原审第三人唐菊芳，女，1960年11月2日出生，汉族，住上海市浦东新区。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市第二中级人民法院", "entity_type": "法院", "entity_idx": [[0, 11]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "唐建忠", "entity_type": "原告", "entity_idx": [[45, 48]]},
                    {"entity_name": "唐杰杰", "entity_type": "原告", "entity_idx": [[87, 98]]},
                    {"entity_name": "陈硕", "entity_type": "原告", "entity_idx": [[129, 131]]},
                    {"entity_name": "唐建国", "entity_type": "原告", "entity_idx": [[169, 172]]},
                    {"entity_name": "唐勇", "entity_type": "原告", "entity_idx": [[210, 212]]},

                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "吴雯麒", "entity_type": "原告委托代理人", "entity_idx": [[383, 386]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "上海市浦东新区北蔡镇人民政府", "entity_type": "被告", "entity_idx": [[410, 424]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "顾娇娇", "entity_type": "被告委托代理人", "entity_idx": [[505, 508]]},
                    {"entity_name": "郑方优", "entity_type": "被告委托代理人", "entity_idx": [[536, 539]]},
                ]}
            ],
            "id": 8},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海金融法院 民事判决书 案号：（2022）沪74民终396号 上诉人（原审被告）：陈军，男，1973年11月18日出生，汉族，住江苏省连云港市海州区。 "
                    "被上诉人（原审原告）：上海唯渡网络科技有限公司，住所地上海市长宁区镇宁路465弄161号4号楼116室。 法定代表人：武文秀，总经理。 委托诉讼代理人：王颖，上海瀛泰律师事务所律师。 "
                    "委托诉讼代理人：段晨澜，上海瀛泰律师事务所律师。 原审被告：王东，男，1971年11月16日出生，汉族，住江苏省灌南县。 "
                    "原审被告：李全浦，男，1966年8月8日出生，汉族，住江苏省连云港市赣榆区。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海金融法院", "entity_type": "法院", "entity_idx": [[0, 6]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "上海唯渡网络科技有限公司", "entity_type": "原告", "entity_idx": [[88,100]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "王颖", "entity_type": "原告委托代理人", "entity_idx": [[153, 155]]},
                    {"entity_name": "段晨澜", "entity_type": "原告委托代理人", "entity_idx": [[177, 180]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "陈军", "entity_type": "被告", "entity_idx": [[42, 44]]},
                    {"entity_name": "王东", "entity_type": "被告", "entity_idx": [[199, 201]]},
                    {"entity_name": "李全浦", "entity_type": "被告", "entity_idx": [[235, 238]]},
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": []}
            ],
            "id": 9},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市第二中级人民法院 民事判决书 案号：（2022）沪02民终4311号 上诉人（原审原告）：张伯毅，男，1949年3月12日出生，汉族，户籍所在地上海市。 "
                    "上诉人（原审原告）：金立彩，女，1952年12月5日出生，汉族，户籍所在地上海市。 上诉人（原审原告）：张某1，男，1982年9月23日出生，汉族，户籍所在地上海市。 "
                    "上诉人（原审原告）：XXX，女，1982年8月22日出生，汉族，户籍所在地上海市。 上诉人（原审原告）：张某2，男，2008年9月25日出生，汉族，户籍所在地上海市。 "
                    "法定代理人：张某1（系张某2之父），男，户籍所在地上海市。 法定代理人：XXX（系张某2之母），女，户籍所在地上海市。 上列五上诉人共同委托诉讼代理人：孙丹毅，上海汉盛律师事务所律师。 "
                    "上诉人（原审被告）：张志毅，男，1956年4月16日出生，汉族，户籍所在地上海市。 委托诉讼代理人：栾伟强，上海劲力律师事务所律师。 委托诉讼代理人：高圳南，上海劲力律师事务所律师。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市第二中级人民法院", "entity_type": "法院", "entity_idx": [[0, 11]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "张伯毅", "entity_type": "原告", "entity_idx": [[48, 51]]},
                    {"entity_name": "金立彩", "entity_type": "原告", "entity_idx": [[90, 93]]},
                    {"entity_name": "张某1", "entity_type": "原告", "entity_idx": [[132, 135]]},
                    {"entity_name": "XXX", "entity_type": "原告", "entity_idx": [[174, 177]]},
                    {"entity_name": "张某2", "entity_type": "原告", "entity_idx": [[216, 219]]},

                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "孙丹毅", "entity_type": "原告委托代理人", "entity_idx": [[324, 327]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "张志毅", "entity_type": "被告", "entity_idx": [[351, 354]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "栾伟强", "entity_type": "被告委托代理人", "entity_idx": [[391, 394]]},
                    {"entity_name": "高圳南", "entity_type": "被告委托代理人", "entity_idx": [[416, 419]]},
                ]}
            ],
            "id": 10},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市第二中级人民法院 民事判决书 案号：（2022）沪02民终4482号 上诉人（原审被告）：上海市警华农业信息科技有限公司，住所地上海市青浦区徐泾镇振泾路198号1幢2层A区175室。 "
                    "法定代表人：徐金龙，总经理。 被上诉人（原审原告）：上海宏牛建设发展有限公司，住所地上海市青浦区公园路99号舜浦大厦7层A区748室。 法定代表人：田印芬，执行董事兼总经理。 "
                    "委托诉讼代理人：孙慧凌，北京观韬中茂（上海）律师事务所律师。 委托诉讼代理人：周楠，北京观韬中茂（上海）律师事务所律师。 "
                    "原审第三人：上海天时印刷有限公司，住所地上海市青浦区沪青平公路4501弄2号。 法定代表人：张小玲，董事长。 委托诉讼代理人：蒋西旺，北京惠诚律师事务所上海分所律师。 "
                    "原审第三人：上海市青浦区人民政府夏阳街道办事处，住所地上海市青浦区外青松公路6300号。 法定代表人：徐川，主任。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市第二中级人民法院", "entity_type": "法院", "entity_idx": [[0, 11]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "上海宏牛建设发展有限公司", "entity_type": "原告", "entity_idx": [[121, 133]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "孙慧凌", "entity_type": "原告委托代理人", "entity_idx": [[191, 194]]},
                    {"entity_name": "周楠", "entity_type": "原告委托代理人", "entity_idx": [[222, 224]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "上海市警华农业信息科技有限公司", "entity_type": "被告", "entity_idx": [[48, 63]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": []}
            ],
            "id": 11},
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海金融法院 民事判决书 案号：（2022）沪74民终493号 上诉人（原审原告）：深圳市优信鹏达二手车经纪有限公司嘉兴第一分公司，营业场所浙江省嘉兴市南湖区文贤路1851号汽车商贸园二手车市场2"
                    "号楼303室。 负责人：王桐。 委托诉讼代理人：徐凯佩，上海易锦律师事务所律师。 "
                    "被上诉人（原审被告）：中国人寿财产保险股份有限公司嘉兴中心支公司，营业场所浙江省嘉兴市经济技术开发区秦逸路14号、18号、22号，华隆广场1幢201-202室、204室、301-305室、401"
                    "-404室、407-410室、501-510室。 负责人：黄渊，总经理。 委托诉讼代理人：吕栋晓，男，公司员工。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海金融法院", "entity_type": "法院", "entity_idx": [[0, 6]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "深圳市优信鹏达二手车经纪有限公司嘉兴第一分公司", "entity_type": "原告",
                     "entity_idx": [[42, 65]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "徐凯佩", "entity_type": "原告委托代理人", "entity_idx": [[122, 125]]},
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "中国人寿财产保险股份有限公司嘉兴中心支公司", "entity_type": "被告",
                     "entity_idx": [[150, 171]]}
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "吕栋晓", "entity_type": "被告委托代理人", "entity_idx": [[281, 284]]}
                ]}
            ],
            "id": 12}
    ]
    dev_data = [
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海市第二中级人民法院 民事判决书 案号：（2022）沪02民终4299号 上诉人（原审原告）：李荣芳，女，1957年7月18日出生，汉族，户籍所在地上海市。 "
                    "委托诉讼代理人：蒋新中，上海正贯长虹律师事务所律师。 被上诉人（原审被告）：厉国义，男，1958年1月25日出生，汉族，户籍所在地上海市。 "
                    "被上诉人（原审被告）：厉永智，男，1991年3月18日出生，汉族，户籍所在地上海市。 上列两被上诉人共同委托诉讼代理人：李杨，上海市天一律师事务所律师。",
            "choices": [
                {"entity_type": "法院", "label": 0, "entity_list": [
                    {"entity_name": "上海市第二中级人民法院", "entity_type": "法院", "entity_idx": [[0, 11]]}
                ]},
                {"entity_type": "原告", "label": 0, "entity_list": [
                    {"entity_name": "李荣芳", "entity_type": "原告", "entity_idx": [[48, 51]]}
                ]},
                {"entity_type": "原告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "蒋新中", "entity_type": "原告委托代理人", "entity_idx": [[88, 91]]}
                ]},
                {"entity_type": "被告", "label": 0, "entity_list": [
                    {"entity_name": "厉国义", "entity_type": "被告", "entity_idx": [[118, 121]]},
                    {"entity_name": "厉永智", "entity_type": "被告", "entity_idx": [[161, 164]]},
                ]},
                {"entity_type": "被告委托代理人", "label": 0, "entity_list": [
                    {"entity_name": "李杨", "entity_type": "被告委托代理人", "entity_idx": [[210, 212]]}
                ]}
            ],
            "id": 0},
    ]
    test_data = [
        {
            "task_type": "抽取任务",
            "subtask_type": "实体识别",
            "text": "上海金融法院"
                    "民事判决书"
                    "案号：（2022）沪74民终186号"
                    "上诉人（原审被告）：潢川县发展投资有限责任公司，住所地河南省信阳市潢川县财政局。"
                    "法定代表人：梅利平，董事长。"
                    "委托诉讼代理人：李明修，该公司员工。"
                    "委托诉讼代理人：李阳，河南捷达律师事务所律师。"
                    "被上诉人（原审原告）：海发宝诚融资租赁有限公司（原名中远海运租赁有限公司），住所地中国（上海）自由贸易试验区福山路450号3E室。"
                    "法定代表人：陈易明，总经理。"
                    "委托诉讼代理人：邹华恩，北京德和衡（上海）律师事务所律师。"
                    "委托诉讼代理人：王坤，北京德和衡（上海）律师事务所律师。",
            "choices": [
                {"entity_type": "法院"},
                {"entity_type": "原告"},
                {"entity_type": "原告委托代理人"},
                {"entity_type": "被告"},
                {"entity_type": "被告委托代理人"}
            ],
            "id": 0}
    ]

    model = UbertPiplines(args)
    model.fit(train_data, dev_data)
    result = model.predict(test_data)
    for line in result:
        pprint(line)


if __name__ == "__main__":
    main()