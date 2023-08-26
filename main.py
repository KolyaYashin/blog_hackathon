from flask import Flask,request,render_template, redirect


app = Flask(__name__)

def model(fname, platform):
    if platform=='telegram':
        return 1
    if platform=='vk':
        return 2
    if platform=='youtube':
        return 3
    else:
        return 4


@app.route('/')
def start():

    return render_template('index.html')


if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)