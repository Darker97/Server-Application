import flask
import os
from flask import Flask, flash, request, redirect, url_for, send_file, Response
from werkzeug.utils import secure_filename
from DarknetFunction import getPaintedImage, getTable, getBoxes
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True

UPLOAD_FOLDER = 'UploadSave/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/hi')
def hello_world():
    return 'Hello, World!'

@app.route('/getBoxes/', methods=['GET', 'POST'])
def upload_file():
    TempName = ""

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            TempName = filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file', filename=filename))

    ReturnString = json.dumps(getBoxes(UPLOAD_FOLDER + TempName))

    return ReturnString

@app.route('/getPainted/', methods = ['GET', 'POST'])
def secondUpload():
    TempName = ""

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            TempName = filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file', filename=filename))

    Answer = getPaintedImage(UPLOAD_FOLDER + TempName)
    file = open(Answer)
    return send_file(file, mimetype='image/jpg')

@app.route('/getTable/', methods = ['GET', 'POST'])
def thirdUpload():
    TempName = ""

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            TempName = filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file', filename=filename))

    Answer = getTable(UPLOAD_FOLDER + TempName)
    
    if Answer is 0:
        return "No table found"

    file = open("result.jpg")
    return send_file(file, mimetype='image/jpg')

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

app.debug = True
app.run()