import cv2
import os
import base64
import time
import numpy as np
import argparse
import io
from time import sleep
from PIL import Image
from wide_resnet import WideResNet
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from acquisitionKinect import AcquisitionKinect
from frame import Frame
from keras.utils.data_utils import get_file
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from yolo_utils import infer_image, show_image
from keras.utils.data_utils import get_file
from keras_yolo3.yolo import YOLO
from Utils.utils import  detect_object
import colorsys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.models import load_model
from keras.layers import Input
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess
from flask import Flask, render_template, send_from_directory, Response ,jsonify
import threading
from imutils.video import VideoStream
import imutils
from flask_cors import CORS

outputFrame = None
lock = threading.Lock()

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

class YOLO2(object):
    _defaults = {
        "model_path": 'model_weights/object_trained_weights.h5',
        #"model_path": 'model_data/yolo_weights.h5',
        "anchors_path": 'model_weights/yolo_anchors.txt',
        "classes_path": 'model_classes/object_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "text_size": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],  # [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 1000
        fontScale = 1
        ObjectsList = []

        label = ""
        score = ""
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{}'.format(predicted_class)
            # label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom - top) / 2 + top
            mid_v = (right - left) / 2 + left
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2

            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], 2)

            # get text size
            size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            #(test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                                #  thickness / self.text_size, 2)

            # put text rectangle
            cv2.rectangle(image, (left, top - size[1]), (left + size[0], top), self.colors[c],
                          thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (left, top - 2), font, font_scale,(255, 255, 255), thickness)

            # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return label ,score

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        # image = cv2.imread(image, cv2.IMREAD_COLOR)
        #original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        label , score = self.detect_image(image)
        return label , score

class FaceCV(object):
    """
    Singleton class for face recongnition task
    """

    CASE_PATH = ".\\harcascade\\haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = ".\\model_weights\\weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "model_weights").replace("//", "\\")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)
        self.model._make_predict_function()

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)


    def detect_face(self,frame):

        age = 'Detecting...'
        gender = 'Detecting...'
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        with session1.as_default():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(self.face_size, self.face_size)
            )
            if faces is not():

                # placeholder for cropped faces
                face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
                for i, face in enumerate(faces):
                    face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                    (x, y, w, h) = cropped
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    face_imgs[i, :, :, :] = face_img

                if len(face_imgs) > 0:
                    # predict ages and genders of the detected faces
                    results = self.model.predict(face_imgs)
                    predicted_genders = results[0]
                    ages = np.arange(0, 101).reshape(101, 1)
                    predicted_ages = results[1].dot(ages).flatten()

                # draw results
                for i, face in enumerate(faces):
                    label = "{}, {}".format(int(predicted_ages[i]-3),
                                            "F" if predicted_genders[i][0] > 0.5 else "M")

                    self.draw_label(frame, (face[0], face[1]), label)
                    print(int(predicted_ages[i]))

                    if(predicted_genders[i][0] > 0.5):
                        gender = "Female"
                    else:
                        gender = "Male"

                    age = (int(predicted_ages[i])-3)

        return age , gender

def get_args():


    ageparser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ageparser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    ageparser.add_argument("--width", type=int, default=8,
                        help="width of network")

    ageargs = ageparser.parse_args()

    return ageargs

def my_Obj(frame):

    myObject = 'Detecting...'
    with g3.as_default():
        label, score = yolo.detect_img(frame)
        myObject = label
    return myObject


def detect_emotions(frame):
    emotion = 'Detecting...'
    with session2.as_default():
        # OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
        #image = cv2.cvtColor(frm, cv2.COLOR_RGBA2RGB)

        #colorimage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad",
                        6: "Surprised"}

        facecasc = cv2.CascadeClassifier('harcascade/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            print(emotion_dict[maxindex])
            print(np.argmax(prediction))

            emotion = emotion_dict[maxindex]

        return emotion
            # show the output image
        #cv2.imshow('Video', cv2.resize(image, (1600, 960), interpolation=cv2.INTER_CUBIC))
       # if cv2.waitKey(1) & 0xFF == ord('q'):
           # break

def load_emotions_model():
    global model
    # Define data generators
    train_dir = 'data/train'
    val_dir = 'data/test'

    num_train = 28709
    num_val = 7178
    batch_size = 64
    num_epoch = 50

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('model_weights/emotions_model.h5')
    model._make_predict_function()

def detection():
    global vs, outputFrame, lock
    global themAge,themGender
    global themEmo,themObj
    global themImage


    kinect = AcquisitionKinect()
    frame2 = Frame()

    # initialize the motion detector and the total number of frames
    # read thus far
    #cap = cv2.VideoCapture(0)
    check = ""
    while True:
        kinect.get_frame(frame2)
        kinect.get_color_frame()
        frm = kinect._frameRGB
        #ret, frm = cap.read()

        # OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
        ##gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (7, 7), 0)

        image = cv2.cvtColor(frm, cv2.COLOR_RGBA2RGB)

        age , gender = face.detect_face(image)
        emotion = detect_emotions(image)
        obj = my_Obj(image)

        if(age != "Detecting..." and gender != "Detecting..."):
            themAge = age
            themGender = gender
        else:
            themAge = ""
            themGender = ""

        if(emotion != "Detecting..."):
            themEmo = emotion
        else:
            themEmo = ""

        if(obj != "Detecting..."):
            themObj = obj
        else:
            themObj = ""

        #cv2.imshow('Video', cv2.resize(image, (1600, 960), interpolation=cv2.INTER_CUBIC))
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

        with lock:
            outputFrame = image

            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            themImage = base64.b64encode(cv2.imencode('.jpg', outputFrame)[1]).decode()
            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(detection(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/getValues')
def getValues():
    return jsonify({'age':themAge,'gender':themGender,'emotion':themEmo,'object':themObj,"image":themImage})



def main():
    global face
    global session1
    global session2
    global net
    global g3
    global g4
    global yolo
    global FLAGS2

    args = get_args()
    depth = args.depth
    width = args.width

    g1 = tf.Graph()
    g2 = tf.Graph()

    session1 = tf.Session(graph=g1)
    session2 = tf.Session(graph=g2)

    with session1.as_default():
      with g1.as_default():
          face = FaceCV(depth=depth, width=width)

    with session2.as_default():
        with g2.as_default():
            load_emotions_model()

    #detect_emotions()

   # net = load_object_yolo()

   # g4 = tf.get_default_graph()
    yolo = YOLO2()

    g3 = tf.get_default_graph()

    app.run()
    #detect_object()
    #face.detect_face()

    #t = threading.Thread(target=detection())
    #t.daemon = True
    #t.start()

    #detection()

    #myObj()

if __name__ == '__main__':
    main()
    #detection()