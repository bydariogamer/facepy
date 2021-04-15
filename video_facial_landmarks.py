# imports
import dlib
import cv2
import sys
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import pyvirtualcam
import numpy
import random
from particles import Particle


TEST = True

print('LOADING DETECTOR')
detector = dlib.get_frontal_face_detector()
print('DETECTOR LOADED\n')

print('LAODING MODEL')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
print('MODEL LOADED\n')

# video stream
vs = VideoStream().start()

# our image size
size = 400

# create a virtual cam
cam = pyvirtualcam.Camera(width=size, height=size, fps=20)

# create an empty list for face dots so that script doesn't crash if there's no face
face = [(0, 0) for _ in range(68)]

# empty list of particles
HAIR = []

# main video loop
stream = True
print('STREAMING BEGINS')
while stream:

    pic = vs.read()     # take a new frame from camera
    pic = imutils.resize(pic, width=size)   # resize image to sizexsize
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)    # convert to grayscale
    
    # detect faces in the grayscale frame
    faces = detector(gray, 0)

    # creates a new frame, completly black
    frame = numpy.zeros((size, size, 3), numpy.uint8)

    # turn the prediction to numpy array and turn it to dots (tuples) stored in a list
    dots = []
    if faces:
        face = face_utils.shape_to_np(predictor(gray, faces[0]))    # I only take first face... there might be more
    for (x, y) in face:
        dots.append((x, y))

    """
    # draw dots to the frame
    for dot in dots:
        cv2.circle(frame, dot, 2, (0, 0, 255), -1)
    """

    # parts of the face
    NOSE = numpy.array([dots[30], dots[31], dots[33], dots[35]])
    L_NOSE = numpy.array([dots[28], dots[30], dots[35]])
    R_NOSE = numpy.array([dots[28], dots[30], dots[31]])
    RR_FACE = numpy.array([dots[0], dots[3], dots[6]])
    R_FACE = numpy.array([dots[0], dots[6], dots[8], dots[27], dots[21], dots[17]])
    LL_FACE = numpy.array([dots[10], dots[13], dots[16]])
    L_FACE = numpy.array([dots[16], dots[10], dots[8], dots[27], dots[22], dots[26]])
    R_EYE = numpy.array([dots[36], dots[38], dots[39], dots[40]])
    L_EYE = numpy.array([dots[45], dots[43], dots[42], dots[47]])
    MOUTH = numpy.array([dots[48], dots[51], dots[54], dots[66]])

    # particles
    for dot in dots[17:28]:
        r = random.uniform(-20, 20)
        color = 211 + r, 34 + r, 19 + r
        new_dot = dot[0], dot[1] + (dots[19][1] - dots[41][1])
        HAIR.append(Particle(new_dot, color, (int(random.uniform(-10, 10)), 30)))
    if len(HAIR) > 40:
        HAIR = HAIR[-40:-1]

    # draw a face
    cv2.fillPoly(frame, [RR_FACE], (30, 30, 250), cv2.LINE_AA)
    cv2.fillPoly(frame, [R_FACE], (0, 0, 250), cv2.LINE_AA)
    cv2.fillPoly(frame, [LL_FACE], (200, 200, 30), cv2.LINE_AA)
    cv2.fillPoly(frame, [L_FACE], (250, 250, 0), cv2.LINE_AA)
    cv2.fillPoly(frame, [NOSE], (200, 200, 200), cv2.LINE_AA)
    cv2.fillPoly(frame, [R_NOSE], (250, 250, 250), cv2.LINE_AA)
    cv2.fillPoly(frame, [L_NOSE], (180, 180, 180), cv2.LINE_AA)
    cv2.fillPoly(frame, [R_EYE], (55, 255, 255), cv2.LINE_AA)
    cv2.fillPoly(frame, [L_EYE], (50, 250, 250), cv2.LINE_AA)
    cv2.fillPoly(frame, [MOUTH], (150, 20, 20), cv2.LINE_AA)

    for index, particle in enumerate(HAIR):
        particle.draw(frame)
        particle.update()
        if particle.delete:
            del HAIR[index]

    print(len(HAIR))
    # show the frame
    cv2.imshow("VIRTUAL FACE", frame[:, :, ::-1])

    # send image to cam
    cam.send(frame)
    cam.sleep_until_next_frame()
 
    # break loop if `Q` is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        stream = False


# cleanup
cv2.destroyAllWindows()
vs.stop()
sys.exit()
