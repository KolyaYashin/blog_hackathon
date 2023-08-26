from flask import Flask,request,render_template, redirect
from werkzeug.utils import secure_filename
import os

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


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def start():
    return render_template('index.html',result='Здесь будет выводиться результат после обработки фотографии и ID.')

def allowed_file(fname:str):
    if fname.endswith(tuple(ALLOWED_EXTENSIONS)):
        return 1
    else:
        return 0

def post():
    platform = request.form['platform']
    id = request.form['id']
    if id=='':
        return render_template('index.html', result = 'Вы не ввели ID')

    id = int(id)
    file = request.files['photo']
    print(file, type(file))

    if file.filename == '':
        return render_template('index.html', result = 'Вы не выбрали фотографию')

    if file and allowed_file(file.filename):
        filename = str(id)+"_"+secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

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