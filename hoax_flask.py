#! /home/easysoft/anaconda3/envs/hoax/bin/python

from hoax import load_mdl, main
from flask import Flask, request

app = Flask(__name__)

@app.before_first_request
def load_model_to_app():
    app.model = load_mdl('hoax_data.pkl')

@app.route('/hoax', methods=['GET','POST'])
def hoax_or_na():
    if request.method == 'POST':
        import time
        st = time.time()
        f = request.form['text']
        val = main(f,app.model)
        en = time.time()
        el = en - st
        with open('/home/easysoft/output_hoax.txt','a') as ou:
            print(el,file=ou)
        return "{}".format(val)
    else:
        if 'e' in request.args:
            return request.args['e']
        return "Hello World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8092, debug=False)