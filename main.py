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
import numpy as np
import cv2

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


def get_file(file_name, subs, platform):
    spl=file_name.split("_")
    id=spl[0]
    now = datetime.now()
    with open(f'table_{platform}.csv', 'a') as f_object:
        writer_object = writer(f_object)

        results = pd.read_csv(f'table_{platform}.csv')
        row_count = len(results)
        List = [row_count, id, now, subs]
        writer_object.writerow(List)

        f_object.close()


class_model = load_model('blog_model/')

def photo_class(file_path, model=class_model):
    im = cv2.imread(file_path)
    im=cv2.resize(im,(256,256))
    names = {0:'tg',1:'vk',2:'yt',3:'zn'}
    im = np.array([im])
    print(names[np.argmax(np.array(model(im)))])
    return names[np.argmax(np.array(model(im)))]


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
    id = request.form['id']
    if id=='':
        return render_template('index.html', result = 'Вы не ввели ID')

    id = int(id)
    file = request.files['photo']

    if file.filename == '':
        return render_template('index.html', result = 'Вы не выбрали фотографию')

    if file and allowed_file(file.filename):
        filename = str(id)+"_"+secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
    else:
        return render_template('index.html', result='Плохой формат файла')


    platform = photo_class(path)
    global result2show
    result2show = get_count_subs(path,type_platform=platform)
    if result2show is None:
        result2show = 'Загрузите другую фотографию'
    if not result2show == 'Загрузите другую фотографию':
        get_file(filename, int(result2show),platform=platform)
    os.remove(path)
    return render_template('index.html', result=result2show)

@app.route('/', methods=['POST'])
def post_home():
    return post()

@app.route('/upload', methods=['GET'])
def get():
    platform = platform_convertor(request.url.split('=')[-1])

    return send_file(
        f'table_{platform}.csv',
        mimetype='text/csv',
        download_name=f'table_{platform}.csv',
        as_attachment=True
    )

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)