from pprint import pprint

from paddle_serving_server.pipeline import PipelineClient
from numpy import array, float32, int32, float64

import numpy as np
import json

def print_ret(outputs, texts, schema):
    for text, output in zip(texts, outputs):
        print("1. Input text: ")
        print(text)
        print("2. Input schema: ")
        print(schema)
        print("3. Result: ")
        pprint(output)
        print("-----------------------------")

def init_client():
    client = PipelineClient()
    client.connect(['127.0.0.1:18090'])
    return client

def test_demo(client):
    texts = [
        '"北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"',
        '原告赵六，2022年5月29日生\n委托代理人孙七，深圳市C律师事务所律师。\n被告周八，1990年7月28日出生\n委托代理人吴九，山东D律师事务所律师'
    ]
    schema = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]

    ret = client.predict(feed_dict={"tokens": text1})
    value = json.loads(ret.value[0])
    print_ret(value, texts, schema)

if __name__ == "__main__":
    client = init_client()
    test_demo(client)