from __future__ import division
import caffe
import sys
#import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
from PIL import Image
from imutils.video import VideoStream

caffe.set_mode_gpu()

#model path
MODEL_JOB_DIR = './models'
DATASET_JOB_DIR = './models'
ARCHITECTURE = MODEL_JOB_DIR + '/deploy.prototxt'
WEIGHTS = MODEL_JOB_DIR + '/snapshot_iter_735.caffemodel'

#caffe network
net = caffe.Classifier(ARCHITECTURE, WEIGHTS,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256)) 
labels = ['cat','dog']

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # rgb = imutils.resize(frame, width=750)
    test_image = cv2.resize(frame, (256,256))/255
    #test_image = cv2.resize(rgb_small_frame,(256,256))

    #img=Image.open('/root/SharDir/20190816-084140-3ab8_epoch_30.0/img/1.jpg')
    #test_image = cv2.resize(img, (28,28))
    mean_image = caffe.io.load_image(DATASET_JOB_DIR + '/mean.jpg')
    test_image = test_image-mean_image
    start_time = time.time()
    prediction = net.predict([test_image])
    total_time = time.time() -start_time
    #print("time: {0:.5f} sec".format(total_time))
    description = 'output label:{} , {} %'.format(labels[prediction.argmax()],prediction[0].max()*100)
    #print 'output label:', labels[prediction[0].argmax()]
    #print(prediction)
    cv2.putText(frame, description, (10, 40), cv2.FONT_ITALIC,0.75, (0, 255, 0), 2)
    
    #display(frame)
    cv2.imshow("Frame", frame)
    keyCode = cv2.waitKey(30) & 0xFF
    if keyCode == 27:
        break

    #do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
