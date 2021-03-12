import webbrowser

from flask import Flask, render_template, Response, jsonify, request
import sys
import face_recognition
import cv2
import numpy as np
import datetime
from werkzeug import secure_filename
import pandas as pd
import requests, json
import subprocess
import sys
from re import findall
import base64
from glob import glob
import os

data = pd.read_csv('static/EncodedFaces.csv', error_bad_lines=False)
face_names = []
frame = []
nItem = []
nameDic = {}

lis = []
known_face_names = []
from glob import glob


def text_file():
    global filename
    filename = "access_token.txt"
    file_exists = os.path.isfile(filename)
    if file_exists:
        print("Access Token file Exists")
    else:
        f = open(filename, "w")
        print("New access file create")


def train():
    global lis
    global known_face_names
    filenames = glob('static/datfile/*.dat')
    lis = []
    known_face_names = []
    for i in filenames:
        k = str(i)[7:]
        known_face_names.append(k[:-4])
        lis.append(np.fromfile(i, dtype=float))


# c = np.fromfile('static/test2.dat', dtype=float)
# dataframes = [pd.read_csv(f) for f in filenames]
train()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


def get_frame():
    global lis
    global known_face_names
    # global lis
    video_capture = cv2.VideoCapture(0)
    # video_capture =cv2.VideoCapture('rtsp://admin:admin123@192.168.10.108:554')

    known_face_encodings = lis
    

    face_locations = []
    face_encodings = []

    global face_names
    global frame
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()
        scale_percent = 300
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                for i in range(2):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        if face_distances[0] > 0.45:
                            known_face_encodings_copy = known_face_encodings[0]
                            known_face_encodings[0] = known_face_encodings[best_match_index]
                            known_face_encodings[best_match_index] = known_face_encodings_copy
                            known_face_names_id_copy = known_face_names[0]
                            known_face_names[0] = known_face_names[best_match_index]
                            known_face_names[best_match_index] = known_face_names_id_copy
                            name = "Unknown"
                        else:
                            name = known_face_names[best_match_index]

                face_names.append(name)
        process_this_frame = not process_this_frame

        imgencode = cv2.imencode('.jpg', frame)[1]
        stringData = imgencode.tostring()
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')
    del video_capture


@app.route('/calc')
def calc():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_access_token():
    authorize_url = ""
    token_url = "http://203.190.10.139:8069/api/authentication/oauth2/token"

    # callback url specified when the application was defined

    # client (application) credentials - located at apim.byu.edu
    client_id = 'BackendApplicationFlowDemoClientKey'
    client_secret = 'BackendApplicationFlowDemoClientSecret'

    authorization_redirect_url = authorize_url + '?response_type=code&client_id=' + client_id + '&redirect_uri=' + '&scope=openid'

    print("go to the following url on the browser and enter the code from the returned url: ")
    print("---  " + authorization_redirect_url + "  ---")
    authorization_code = "uuu"

    # step I, J - turn the authorization code into a access token, etc
    data = {'grant_type': 'client_credentials', 'code': authorization_code}
    print("requesting access token")
    access_token_response = requests.post(token_url, data=data, verify=False, allow_redirects=False,
                                          auth=(client_id, client_secret))

    print("response")
    print(access_token_response.headers)
    print('body: ' + access_token_response.text)

    # we can now use the access_token as much as we want to access protected resources.
    tokens = json.loads(access_token_response.text)

    global access_token
    access_token = tokens['access_token']

    print("access token: " + access_token)
    # filename = "access_token.txt"
    # if os.stat(filename).st_size == 0:
    #     print('File is empty')
    #     file2write = open(filename, 'w')
    #     file2write.write(access_token)
    #     file2write.close()
    text_file()
    if os.stat(filename).st_size == 0:
        print('File is empty')

        file2write = open(filename, 'w')
        file2write.write(access_token)
        file2write.close()
    else:
        print('File is not empty')
        file2write = open(filename, 'r+')
        file2write.truncate(0)
        file2write.write(access_token)
        file2write.close()
        print("New Access Token Generate ")


def apicall(employee_id):
    print("employee_id")
    employee_id = employee_id.replace("datfile\\", "")
    # callback url specified when the application was defined
    test_api_url = "http://203.190.10.139:8069/attendance/face/" + employee_id

    generate_access_token()
    f = open(filename, "r")
    new_token = f.read()

    api_call_headers = {'Authorization': 'Bearer ' + new_token}
    api_call_response = requests.post(test_api_url, headers=api_call_headers, json={}, verify=False)

    print(api_call_response.text)


@app.route('/price')
def price():
    global data
    global nameDic
    now = datetime.datetime.now()
    for name in face_names:

        if name == "Unknown":
            print("Unknown")
        elif name not in nItem:
            nameDic[name] = now
            nItem.append(name)
            new_row = {'Name': name, 'InTime': now}
            data = data.append(new_row, ignore_index=True)
            data.to_csv('static/EncodedFaces.csv', index=False)
            apicall(name)
        # playsound('static/ring.mp3')
        # GPIO.output(19, GPIO.LOW)
        # GPIO.output(13, GPIO.HIGH)
        else:
            duration = nameDic[name] - now
            duration = abs(duration.total_seconds())
            # GPIO.output(19, GPIO.LOW)
            # GPIO.output(13, GPIO.HIGH)
            if duration > 1000:
                new_row = {'Name': name, 'InTime': nameDic[name], 'OutTime': now}
                data = data.append(new_row, ignore_index=True)
                data.to_csv('static/EncodedFaces.csv', index=False)
                nameDic[name] = now
                apicall(name)
            else:
                pass
    return jsonify({'name': face_names, 'datetime': now})


@app.route('/upload')
def upload():
    return render_template('upload.html')


def getEmployee():
    test_api_url = "http://203.190.10.139:8069/attendance/face/get_employees"

    generate_access_token()
    f = open(filename, "r")
    token = f.read()
    api_call_headers = {'Authorization': 'Bearer ' + token}
    api_call_response = requests.get(test_api_url, headers=api_call_headers, verify=False)

    image_list = []
    text = api_call_response.text

    text = text.replace(": b", "(")
    text = text.replace(", 'is_pull_face_attendance", ")")

    match = findall(r'\(.*?\)', text)

    i = 0

    while i < len(match):
        x = text.replace(match[i], "")
        text = x

        image_base_64 = match[i].replace("('", "")
        image_base_64 = image_base_64.replace("')", "")
        image_list.append(image_base_64)
        i += 1
    # print(text)
    x = text.replace("''", "'")
    x = x.replace("\'", "\"")

    x = x.replace("False", '"False"')
    x = x.replace("True", '"True"')

    print(x)
    # x = json.dumps(x)
    y = json.loads(x)
    # print("===========================================")
    # print("I am not working")
    # print(y)
    #
    # print(len(y))
    return image_list, y










# text_inside_paranthesis =  text[text.findall(': b')+1:text.find('is_pull_face_attendance')]
# print(text_inside_paranthesis)


# print(text)
# val = [text.split(': b', 1)[1].split("is_pull_face_attendance")[0]]
# print(len(val))
# i=0
# while i<len(val):
# 	x = text.replace(val[i], "image")
# 	text=x
# 	i+=1
# print(val)
# print(text)


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        image_list, employee_info = getEmployee()
        # image = base64.b64decode(str(base64String))
        img_counter = 0

        for i in image_list:
            image_64_decode = base64.b64decode(i)

            image_result = open('static/imagefile/' + employee_info[img_counter]['employee_id'] + '.png',
                                'wb')  # create a writable image and write the decoding result
            image_result.write(image_64_decode)
            img_counter += 1

        # cv2.imshow(image,"name")
        # images = face_recognition.load_image_file("static/imagefile/111113.png")

        images = [cv2.imread(file) for file in glob('static/imagefile/*.png')]
        print(len(images))
        image_counter = 0
        for image in images:
            print((os.listdir('static/imagefile')[image_counter]))
            employee_id = (os.listdir('static/imagefile')[image_counter]).replace(".png", "")
            print(employee_id)
            # name = face_recognition.load_image_file(f.filename)
            encoding = face_recognition.face_encodings(image)[0]
            encoding.tofile('static/datfile/' + employee_id + '.dat')
            image_counter += 1

        print(type(os.listdir('static/imagefile')[0]))

        train()
        return 'file uploaded successfully'


def main():
    # The reloader has not yet run - open the browser
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://localhost:5000/')

    # Otherwise, continue as normal
    app.run(host="localhost", port=5000)


if __name__ == '__main__':
    main()
