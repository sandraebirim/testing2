# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from adafruit_servokit import ServoKit
from centroidtracker import CentroidTracker
#from singlemotiondetector import SingleMotionDetector
from imutils.video import WebcamVideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

kit = ServoKit(channels=16)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
args = vars(ap.parse_args())

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)


# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])



# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = WebcamVideoStream(src=-1).start()
time.sleep(2.0)

def my_map(val, in_min, in_max, out_min, out_max):
	return int((val-in_min)*(out_max-out_min) / (in_max-in_min) + out_min)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion(frameCount,H,W):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
#	md = SingleMotionDetector(accumWeight=0.1)
#	total = 0

	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

# if the frame dimensions are None, grab them
		if W is None or H is None:
			(H, W) = frame.shape[:2]
		center = int(W/2)
	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
		blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		rects = []

	# loop over the detections
		for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
			if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
				(startX, startY, endX, endY) = box.astype("int")
				cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
		objects = ct.update(rects)


	# loop over the tracked objects
		value = 0
		for (objectID, centroid) in objects.items():
			if objectID == 0:	
		# draw both the ID of the object and the centroid of the
		# object on the output frame
				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
				#center coordinates, centroid [0], centroid[1]
				value = centroid[0]
	# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		angle = my_map(value, 0, 500, 0, 180)
		print(angle)
		for i in range(len(kit.servo)):
			kit.servo[i].angle = angle
		time.sleep(5)
#		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
		"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
	#	if total > frameCount:
			# detect motion in the image
	#		motion = md.detect(gray)

			# cehck to see if motion was found in the frame
	#		if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
	#			(thresh, (minX, minY, maxX, maxY)) = motion
	#			cv2.rectangle(frame, (minX, minY), (maxX, maxY),
	#				(0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
	#	md.update(gray)
	#	total += 1

		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()
		
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
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
#	ap = argparse.ArgumentParser()
#	ap.add_argument("-i", "--ip", type=str, required=True,
#		help="ip address of the device")
#	ap.add_argument("-o", "--port", type=int, required=True,
#		help="ephemeral port number of the server (1024 to 65535)")
#	ap.add_argument("-f", "--frame-count", type=int, default=32,
#		help="# of frames used to construct the background model")
#	args = vars(ap.parse_args())
	
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],H,W))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()
