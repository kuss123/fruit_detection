import cv2
import tensorflow as tf
import numpy as np

# frameWidth = 640# CAMERA RESOLUTION
# frameHeight = 480
# brightness = 180
threshold = 0.40  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
##pickle_in = open("model_trained.p", "rb")  ## rb = READ BYTE
##model = pickle.load(pickle_in)
model = tf.keras.models.load_model("CNN.model")


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getClassName(classNo):
    if classNo == 0:
        return 'Banana'
    elif classNo == 1:
        return 'Apple'
    elif classNo == 2:
        return 'Strawberry'
    else:
        return 'none'

while True:
    # READ IMAGE
    success, imgOriginal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOriginal)

    img = cv2.resize(img, (50, 50))
    img = preprocessing(img)
    # cv2.imshow("Processed Image", img)
    img = img.reshape(1, 50, 50, 1)
    cv2.putText(imgOriginal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    classIndex = model.predict_classes(img)
    probabilityValue = np.amax(predictions)
    print(classIndex, probabilityValue)
    if probabilityValue > threshold:
        print(getClassName(classIndex))
        cv2.putText(imgOriginal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
