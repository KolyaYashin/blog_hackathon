from flask import Flask,request,render_template, send_file
from werkzeug.utils import secure_filename
import os
from tesseract import get_count_subs
from csv import writer
from datetime import datetime
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

app = Flask(__name__)

def platform_convertor(platform):
    if platform=='Телеграм':
        return 'tg'
    if platform=='Вк':
        return 'vk'
    if platform=='Ютуб':
        return 'yt'
    else:
        return 'zn'


def get_file(file_name, subs):
    spl=file_name.split("_")
    id=spl[0]
    platform=spl[1]
    now = datetime.now()
    with open('table.csv', 'a') as f_object:
        writer_object = writer(f_object)

        results = pd.read_csv('table.csv')
        row_count = len(results)
        List = [row_count, id, platform, now, subs]
        writer_object.writerow(List)

        f_object.close()


class_model = load_model('saved_model/Model/')



UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','PNG','JPG','JPEG'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def start():
    global result2show
    result2show='Здесь будет выводиться результат после обработки фотографии и ID.'
    return render_template('index.html',result=result2show)

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

    if file.filename == '':
        return render_template('index.html', result = 'Вы не выбрали фотографию')

    if file and allowed_file(file.filename):
        filename = str(id)+"_"+platform+"_"+secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
    else:
        return render_template('index.html', result='Плохой формат файла')
    global result2show
    result2show = get_count_subs(path,type_platform=platform)
    if not result2show == 'Загрузите другую фотографию':
        get_file(filename, int(result2show))
    os.remove(path)
    return render_template('index.html', result=result2show)

@app.route('/', methods=['POST'])
def post_home():
    return post()

@app.route('/upload', methods=['GET'])
def get():
    print('ok')
    return send_file(
        'table.csv',
        mimetype='text/csv',
        download_name='table.csv',
        as_attachment=True
    )

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)