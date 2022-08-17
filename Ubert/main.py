import argparse
from pprint import pprint

from fengshen import UbertPiplines


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser = UbertPiplines.piplines_args(total_parser)
    args = total_parser.parse_args()

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
        ,
        # {
        #     "task_type": "抽取任务",
        #     "subtask_type": "实体识别",
        #     "text": "这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。",
        #     "choices": [
        #         {"entity_type": "小区名字"},
        #         {"entity_type": "岗位职责"}
        #     ],
        #     "id": 0}
        # ,
        # {
        #     "task_type": "抽取任务",
        #     "subtask_type": "实体识别",
        #     "text": "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！",
        #     "choices": [
        #         {"entity_type": "时间"},
        #         {"entity_type": "赛事名称"},
        #         {"entity_type": "选手"}
        #     ],
        #     "id": 1}
        # ,
        # {
        #     "task_type": "抽取任务",
        #     "subtask_type": "实体识别",
        #     "text": "（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。",
        #     "choices": [
        #         {"entity_type": "肿瘤的大小"},
        #         {"entity_type": "肿瘤的个数"},
        #         {"entity_type": "肝癌级别"},
        #         {"entity_type": "脉管内癌栓分级"}
        #     ],
        #     "id": 2}
    ]

    model = UbertPiplines(args)
    result = model.predict(test_data)
    for line in result:
        pprint(line)


if __name__ == "__main__":
    main()
