#! /home/support/anaconda3/envs/hoax/bin/python
import hoax
from hoax import load_mdl, main
from flask import Flask, request

app = Flask(__name__)

@app.before_first_request
def load_model_to_app():
    app.model = load_mdl('hoax_data.pkl')
    app.model_cu = load_mdl('hoax_data_cu.pkl')

@app.route('/hoax', methods=['GET','POST'])
def hoax_or_na():
    if request.method == 'POST':
        import time
        st = time.time()
        hoax.mode = hoax.CODE_IB
        f = request.form['text']
        val = main(f,app.model)
        en = time.time()
        el = en - st
        with open('/home/support/output_hoax.txt','a') as ou:
            print(el,file=ou)
        return "{}".format(val)
    else:
        if 'e' in request.args:
            return request.args['e']
        with open('FORMAT_HOAX','r') as fh:
            return fh.read()

@app.route('/hoax_cu', methods=['GET','POST'])
def hoax_or_na_cu():
    if request.method == 'POST':
        import time
        st = time.time()
        hoax.mode = hoax.CODE_CU
        f = request.form['text']
        val = main(f,app.model_cu)
        en = time.time()
        el = en - st
        with open('/home/support/output_hoaxc.txt','a') as ou:
            print(el,file=ou)
        return "{}".format(val)
    else:
        if 'e' in request.args:
            return request.args['e']
        with open('FORMAT_HOAX','r') as fh:
            return fh.read()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8092, debug=False)