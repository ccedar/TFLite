import threading
import time
import RPi.GPIO as GPIO # for ultrasound
import serial  # for communication with Arduino
import Adafruit_TCS34725 # for RGB sensor
import smbus
from argparse import ArgumentParser

# -------------------------- Exception -----------------------
class WarningUltraSound(Exception):
    def __init__(self):
        super().__init__("Too Close")

# --------------------------Sub Thread class구현-----------------------
# ULTRASOUND
class UltraSound(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        print("ultraSound exe\n")
        global ultraSoundFlag


        print("AkibaTV HC-SR04 Start")

        GPIO.setmode(GPIO.BCM)
        # trig = 2
        trig = 4
        GPIO.setup(trig, GPIO.OUT)
        # echo = 3
        echo = 14
        GPIO.setup(echo, GPIO.IN)

        try:
            while True:
                GPIO.output(trig, False)
                time.sleep(0.5)

                GPIO.output(trig, True)
                time.sleep(0.00001)
                GPIO.output(trig, False)

                # echo이 OFF가 되는 시점을 시작시간으로 설정
                while GPIO.input(echo) == 0:
                    start = time.time()

                # echo번이 ON이 되는 시점을 반사파 수신시간으로 설정
                while GPIO.input(echo) == 1:
                    stop = time.time()

                # 초음파가 되돌아오는 시간차로 거리 계산 
                time_interval = stop - start
                distance = time_interval * 17000
                distance = round(distance, 2)

                print("Distance => ", distance, "cm")

                if distance<25:
                    print("ultraSoundFlag is True\n")
                    ultraSoundFlag = True
                    event.wait()
                    event.clear()
                    ultraSoundFlag = False

        except KeyboardInterrupt:
            GPIO.cleanup()
            print("AkibaTV HC-SR04 End")

# RGB sensor
class RGB(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        print("RGB sensor exe\n")
    
        tcs = Adafruit_TCS34725.TCS34725()
        tcs.set_interrupt(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(32, GPIO.OUT)
        GPIO.setup(36, GPIO.OUT)
        GPIO.setup(38, GPIO.OUT)
        pr = GPIO.PWM(32,50)
        pg = GPIO.PWM(36,50)
        pb = GPIO.PWM(38,50)
        pr.start(20)
        pg.start(20)
        pb.start(20)

        while 1:
            r, g, b, c = tcs.get_raw_data()
            color_temp = Adafruit_TCS34725.calculate_color_temperature(r, g, b)
            lux = Adafruit_TCS34725.calculate_lux(r, g, b)
            print('Color: red={0} green={1} blue={2} clear={3}'.format(r, g, b, c))
            time.sleep(1)

            if((r > b) and (r > g)):
                if(b <5):
                    pr.ChangeDutyCycle(80)
                    pg.ChangeDutyCycle(1)
                    pb.ChangeDutyCycle(1)
                    w="red"
                    print(w)
                elif (r-g <20 and r-b > 10 and r>15):
                    pr.ChangeDutyCycle(80)
                    pg.ChangeDutyCycle(65)
                    pb.ChangeDutyCycle(1)
                    w="yellow"
                    print(w)
            elif((r < b) and (b >5)):
                pr.ChangeDutyCycle(1)
                pg.ChangeDutyCycle(1)
                pb.ChangeDutyCycle(80)
                w="blue"
                print(w)
            elif((r < g) and (b <g) and (g>20)):
                if (b<10):
                    pr.ChangeDutyCycle(1)
                    pg.ChangeDutyCycle(80)
                    pb.ChangeDutyCycle(1)
                    w="green"
                    print(w)
                elif(b>10 and r-g <6):
                    pr.ChangeDutyCycle(60)
                    pg.ChangeDutyCycle(5)
                    pb.ChangeDutyCycle(70)
                    w="purple"
                    print(w)
            elif((r > g) and (b > g) and (r>45)):
                pr.ChangeDutyCycle(0)
                pg.ChangeDutyCycle(70)
                pb.ChangeDutyCycle(70)
                w="cyan"
                print(w)
            elif((r > 100) and (b > 90) and (g> 100)):
                pr.ChangeDutyCycle(0)
                pg.ChangeDutyCycle(70)
                pb.ChangeDutyCycle(70)
                w="white"
                print(w)
            elif((r < 25) and (b <20) and (g < 20)):
                pr.ChangeDutyCycle(0)
                pg.ChangeDutyCycle(0)
                pb.ChangeDutyCycle(0)
                w="black"
                print(w)
            else:
                pr.ChangeDutyCycle(20)
                pg.ChangeDutyCycle(20)
                pb.ChangeDutyCycle(20)

        time.sleep(9)
        tcs.set_interrupt(True)
        tcs.disable()
        pr.stop()
        pg.stop()
        pb.stop()
###################################################################################
# Object Detection
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

class ObjectDetection(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
      # Define and parse input arguments
      parser = ArgumentParser()
      parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                          required=True)
      parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                          default='detect.tflite')
      parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                          default='labelmap.txt')
      parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                          default=0.5)
      parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                          default='1280x720')
      parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                          action='store_true')

      args = parser.parse_args()

      MODEL_NAME = args.modeldir
      GRAPH_NAME = args.graph
      LABELMAP_NAME = args.labels
      min_conf_threshold = float(args.threshold)
      resW, resH = args.resolution.split('x')
      imW, imH = int(resW), int(resH)
      use_TPU = args.edgetpu

      # Import TensorFlow libraries
      # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
      # If using Coral Edge TPU, import the load_delegate library
      pkg = importlib.util.find_spec('tflite_runtime')
      if pkg:
          from tflite_runtime.interpreter import Interpreter
          if use_TPU:
              from tflite_runtime.interpreter import load_delegate
      else:
          from tensorflow.lite.python.interpreter import Interpreter
          if use_TPU:
              from tensorflow.lite.python.interpreter import load_delegate

      # If using Edge TPU, assign filename for Edge TPU model
      if use_TPU:
          # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
          if (GRAPH_NAME == 'detect.tflite'):
              GRAPH_NAME = 'edgetpu.tflite'       

      # Get path to current working directory
      CWD_PATH = os.getcwd()

      # Path to .tflite file, which contains the model that is used for object detection
      PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

      # Path to label map file
      PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

      # Load the label map
      with open(PATH_TO_LABELS, 'r') as f:
          labels = [line.strip() for line in f.readlines()]

      # Have to do a weird fix for label map if using the COCO "starter model" from
      # https://www.tensorflow.org/lite/models/object_detection/overview
      # First label is '???', which has to be removed.
      if labels[0] == '???':
          del(labels[0])

      # Load the Tensorflow Lite model.
      # If using Edge TPU, use special load_delegate argument
      if use_TPU:
          interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                    experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
          print(PATH_TO_CKPT)
      else:
          interpreter = Interpreter(model_path=PATH_TO_CKPT)

      interpreter.allocate_tensors()

      # Get model details
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()
      height = input_details[0]['shape'][1]
      width = input_details[0]['shape'][2]

      floating_model = (input_details[0]['dtype'] == np.float32)

      input_mean = 127.5
      input_std = 127.5

      # Check output layer name to determine if this model was created with TF2 or TF1,
      # because outputs are ordered differently for TF2 and TF1 models
      outname = output_details[0]['name']

      if ('StatefulPartitionedCall' in outname): # This is a TF2 model
          boxes_idx, classes_idx, scores_idx = 1, 3, 0
      else: # This is a TF1 model
          boxes_idx, classes_idx, scores_idx = 0, 1, 2

      # Initialize frame rate calculation
      frame_rate_calc = 1
      freq = cv2.getTickFrequency()

      # Initialize video stream
      videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
      time.sleep(1)

      #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
      while True:

          # Start timer (for calculating frame rate)
          t1 = cv2.getTickCount()

          # Grab frame from video stream
          frame1 = videostream.read()

          # Acquire frame and resize to expected shape [1xHxWx3]
          frame = frame1.copy()
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          frame_resized = cv2.resize(frame_rgb, (width, height))
          input_data = np.expand_dims(frame_resized, axis=0)

          # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
          if floating_model:
              input_data = (np.float32(input_data) - input_mean) / input_std

          # Perform the actual detection by running the model with the image as input
          interpreter.set_tensor(input_details[0]['index'],input_data)
          interpreter.invoke()

          # Retrieve detection results
          boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
          classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
          scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

          # Loop over all detections and draw detection box if confidence is above minimum threshold
          for i in range(len(scores)):
              if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                  # Get bounding box coordinates and draw box
                  # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                  ymin = int(max(1,(boxes[i][0] * imH)))
                  xmin = int(max(1,(boxes[i][1] * imW)))
                  ymax = int(min(imH,(boxes[i][2] * imH)))
                  xmax = int(min(imW,(boxes[i][3] * imW)))
                  
                  cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                  # Draw label
                  object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                  if object_name=='person' :
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    print(label)
                    print('FPS: {0: .2F}'.format(frame_rate_calc))
                  elif object_name=='bed':
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    print(label)
                    print('FPS: {0: .2F}'.format(frame_rate_calc))
        
          # Draw framerate in corner of frame
          #print(label)
          #print('FPS: {0: .2F}'.format(frame_rate_calc))
      #    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
      
          # All the results have been drawn on the frame, so it's time to display it.
      #    cv2.imshow('Object detector', frame)

          # Calculate framerate
          t2 = cv2.getTickCount()
          time1 = (t2-t1)/freq
          frame_rate_calc= 1/time1

          # Press 'q' to quit
          if cv2.waitKey(1) == ord('q'):
              break

      # Clean up
      cv2.destroyAllWindows()
      videostream.stop()


# -------------------------- Main Thread -----------------------

# A지점으로 이동...
def aGo():
    global ultraSoundFlag
    while True:
        if ultraSoundFlag == True:
            #ser.write('0'.encode())
            print("stop")
            time.sleep(3)
            event.set()
        #ser.write('1'.encode())
        print("going")
        time.sleep(0.25)

# 쓰레드 간 조율을 위한 이벤트 객체 생성 
event = threading.Event()

# check the port for USB($ ls /dev/tty*) 
ser = serial.Serial("/dev/ttyUSB0", 9600, 
                     parity=serial.PARITY_NONE,
                     timeout=1) 

# subThread Fork
threads = {}
threads['ultraSound'] = UltraSound()
threads['rgb'] = RGB()
threads['objectDetection'] = ObjectDetection()

# daemon thread
for thread in threads.values():
    thread.daemon = True

# very close between object and camera
ultraSoundFlag = False

# navigation
command = input("Write point where you want to go\n")
if command == 'a' or command == 'A':
    for thread in threads.values():
        thread.start()
    aGo()
