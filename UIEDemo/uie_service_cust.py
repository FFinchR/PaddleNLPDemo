from pprint import pprint

import paddle
from paddlenlp import Taskflow
from flask import Flask, request, jsonify
import traceback
import threading

lock = threading.Lock()

schema = ['法院', {'原告': '原告委托代理人'}, {'被告': '被告委托代理人'}]
ie = Taskflow('information_extraction', schema=schema, task_path='./legal_judgment_checkpoint/model_best', device_id=1)
app = Flask(__name__)


@app.route('/uie', methods=['post'])
def uie():
    body = request.form
    text = str(body.get('text'))
    lock.acquire()
    try:
        res = str(ie(text))
        lock.release()
        return jsonify(res)
    except Exception as e:
        lock.release()
        traceback.print_exc()
        return jsonify(e)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8020, debug=True)
