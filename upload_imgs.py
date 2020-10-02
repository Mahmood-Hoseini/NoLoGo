# import os
# from PIL import Image
# from flask import Flask, flash, redirect, request, url_for
# from flask_cors import CORS
# import base64

# app = Flask(__name__)
# CORS(app)

# @app.route('/', methods=['GET', 'POST'])
# def upload_file() :
# 	data = request.get_json()
# 	encodeImage = u''.join(data['value']).encode('utf-8').strip()
# 	imgdata = base64.b64decode(encodeImage)
# 	with open('./data/img.jpg', 'wb') as fd:
# 		fd.write(imgdata)

	
# 	os.chdir('./yolov5/')
# 	yolo = "python detect.py --source ../data/img.jpg --weights weights/best.pt --output ../data/ --conf-thres 0.2 --save-txt"
# 	os.system(yolo)

# 	return "Hello"


# if __name__ == "__main__" :
# 	app.run()


import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './data/sample_imgs2/145369427.jpg'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


if __name__ == '__main__' :
	app.run()