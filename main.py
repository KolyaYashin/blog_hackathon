from flask import Flask,request,render_template, redirect


app = Flask(__name__)

def model(fname, platform):
    if platform=='Телеграм':
        return 1
    if platform=='Вконтакте':
        return 2
    if platform=='YouTube':
        return 3
    else:
        return 4


@app.route('/')
def start():
    return render_template('index.html',result='Здесь будет выводиться результат после обработки фотографии и ID.')

def post():
    platform = request.form['platform']
    id = request.form['id']
    print(platform)
    print(id)
    result2show = model('test.jpg',platform=platform)
    return render_template('index.html', result=result2show)

@app.route('/', methods=['POST'])
def post_home():
    return post()

'''@app.route('/result', methods=['POST'])
def post_result():
    return post()

@app.route('/result')
def result_start():
    return render_template('index.html')'''

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)