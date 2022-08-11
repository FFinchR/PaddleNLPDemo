# -*- encoding: utf-8 -*-
from pprint import pprint
from paddlenlp import Taskflow


def legal_judgement_ie(content):
    schema = ['法院', {'原告': '原告委托代理人'}, {'被告': '被告委托代理人'}]  # Define the schema for entity extraction
    ie = Taskflow('information_extraction', schema=schema, task_path='./checkpoint/model_best') # task_path='./checkpoint/model_best'
    pprint(ie("上海金融法院"
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

def drug_ie(content):
    schema = ['地点', '时间', '毒品重量', {'涉案人': '涉案人'}, {'涉案人': '毒品'},
              {'涉案人': '涉案人'}]  # Define the schema for entity extraction
    ie = Taskflow('information_extraction', schema=schema)
    pprint(
        ie("2016年6月初某日，被告人陆xx经与吸毒人员范xx事先联系，由范xx驾驶牌号为沪C8XXXX红色奥迪轿车至本区卫清东路XXX号门口，被告人陆xx进入该车内将约0.3克甲基苯丙胺以人民币300元的价格贩卖给范xx"))  # Better print results using pprint


if __name__ == '__main__':
    legal_judgement_ie("")
