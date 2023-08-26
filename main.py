from flask import Flask,request,render_template, redirect
from werkzeug.utils import secure_filename
import os
from tesseract import get_count_subs
from csv import writer
from datetime import datetime
import pandas as pd

app = Flask(__name__)

def platform_convertor(platform):
    if platform=='Телеграм':
        return 'tg'
    if platform=='Вконтакте':
        return 'vk'
    if platform=='YouTube':
        return 'yt'
    else:
        return 'zn'

def get_file(file_name, subs):
    id, platform, name = file_name.split("_")
    now = datetime.now()
    with open('table.csv', 'a') as f_object:
        writer_object = writer(f_object)

        results = pd.read_csv('table.csv')
        row_count = len(results)
        List = [row_count, id, platform, now, subs]
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(List)

        # Close the file object
        f_object.close()


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
    platform = platform_convertor(request.form['platform'])
    id = request.form['id']
    if id=='':
        return render_template('index.html', result = 'Вы не ввели ID')

    id = int(id)
    file = request.files['photo']
    print(file, type(file))

    if file.filename == '':
        return render_template('index.html', result = 'Вы не выбрали фотографию')

    if file and allowed_file(file.filename):
        filename = str(id)+"_"+platform+"_"+secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
    print(path)
    result2show = get_count_subs(path,type_platform=platform)
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