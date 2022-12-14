# -*- encoding: utf-8 -*-
from pprint import pprint, pformat
from paddlenlp import Taskflow


def legal_judgement_ie(content):
    schema = ['法院', {'原告': '原告委托代理人'}, {'被告': '被告委托代理人'}]  # Define the schema for entity extraction
    ie = Taskflow('information_extraction', schema=schema,
                  task_path='./legal_judgment_checkpoint/model_best', device_id=1)  # task_path='./checkpoint/model_best'
    pprint(type(content))
    pprint(content)
    # pprint(ie(content))
    print(pformat(ie(content)))

def drug_ie(content):
    schema = ['地点', '时间', '毒品重量', {'涉案人': ['涉案人', '涉案人', '毒品类型']}]  # Define the schema for entity extraction
    ie = Taskflow('information_extraction', schema=schema, task_path='./drug_checkpoint/model_best')
    pprint(content)
    pprint(ie(content))  # Better print results using pprint

def test_ie(content):
    schema = [{"人物": ["祖籍", "父亲", "出生地", "妻子", "母亲", "国籍", "丈夫", "毕业院校", "身高", "出生日期", "民族"]}, {"企业": ["总部地点", "董事长", "创始人", "注册资本", "成立日期"]}, {"生物": ["目"]}, {"歌曲": ["所属专辑", "歌手", "作词", "作曲"]}, {"国家": ["首都", "官方语言"]}, {"影视作品": ["导演", "出品公司", "制片人", "编剧", "主演", "改编自", "上映时间"]}, {"网络小说": ["连载网站", "主角"]}, {"书籍": ["出版社"]}, {"电视综艺": ["主持人", "嘉宾"]}, {"景点": ["所在城市"]}, {"图书作品": ["作者"]}, {"行政区": ["气候", "面积", "邮政编码", "人口数量"]}, {"机构": ["简称", "成立日期", "占地面积"]}, {"历史人物": ["字", "朝代", "号"]}, {"学科专业": ["修业年限", "专业代码"]}, {"地点": ["海拔"]}, {"测试1": ["测试关系"]}, {"音乐专辑": ["NA"]}]
    ie = Taskflow('information_extraction', schema=schema)
    pprint(content)
    pprint(ie(content))

def test_ie(content):
    schema = ['小区名字', '岗位职责']
    ie = Taskflow('information_extraction', schema=schema)
    pprint(content)
    pprint(ie(content))


if __name__ == '__main__':
    # legal_judgement_ie("上海金融法院"
    #                    "民事判决书"
    #                    "案号：（2022）沪74民终186号"
    #                    "上诉人（原审被告）：潢川县发展投资有限责任公司，住所地河南省信阳市潢川县财政局。"
    #                    "法定代表人：梅利平，董事长。"
    #                    "委托诉讼代理人：李明修，该公司员工。"
    #                    "委托诉讼代理人：李阳，河南捷达律师事务所律师。"
    #                    "被上诉人（原审原告）：海发宝诚融资租赁有限公司（原名中远海运租赁有限公司），住所地中国（上海）自由贸易试验区福山路450号3E室。"
    #                    "法定代表人：陈易明，总经理。"
    #                    "委托诉讼代理人：邹华恩，北京德和衡（上海）律师事务所律师。"
    #                    "委托诉讼代理人：王坤，北京德和衡（上海）律师事务所律师。")
    # with open("./data/CAIL2022_ie/step1_test.json", "r", encoding="utf-8") as f:
    #     raw_examples = f.readlines()
    # line = raw_examples[92]
    # drug_ie(line)
    # test_ie("这也让很多业主据此认为，雅清苑是政府公务员挤对了国家的经适房政策。")
    test_ie("查尔斯·阿兰基斯（Charles Aránguiz），1989年4月17日出生于智利圣地亚哥，智利职业足球运动员，司职中场，效力于德国足球甲级联赛勒沃库森足球俱乐部")

# 上海金融法院 民事判决书 案号：（2022）沪74民终186号 上诉人（原审被告）：潢川县发展投资有限责任公司，住所地河南省信阳市潢川县财政局。 法定代表人：梅利平，董事长。 委托诉讼代理人：李明修，该公司员工。 委托诉讼代理人：李阳，河南捷达律师事务所律师。 被上诉人（原审原告）：海发宝诚融资租赁有限公司（原名中远海运租赁有限公司），住所地中国（上海）自由贸易试验区福山路450号3E室。 法定代表人：陈易明，总经理。 委托诉讼代理人：邹华恩，北京德和衡（上海）律师事务所律师。 委托诉讼代理人：王坤，北京德和衡（上海）律师事务所律师。