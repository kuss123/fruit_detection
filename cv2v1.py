#import cv2
#import tensorflow as tf
#CATEGORIES = ["class0", "class1", "class2", "class3"]
#def prepare(file):
#    IMG_SIZE = 50
#    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
#    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#model = tf.keras.models.load_model("CNN.model")
#image = "test.jpg" #your image path
#prediction = model.predict([image])
#prediction = list(prediction[0])
#print(CATEGORIES[prediction.index(max(prediction))])
"""
import cv2 as cv2

cv2.namedWindow("myWindow")

cap = cv2.VideoCapture(0) #open camera
ret,frame = cap.read() #start streaming

windowWidth=frame.shape[1]
windowHeight=frame.shape[0]
print(windowWidth)
print(windowHeight)

cv2.waitKey(0) #wait for a key
cap.release() # Destroys the capture object
cv2.destroyAllWindows() # Destroys all the windows"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, 0)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()