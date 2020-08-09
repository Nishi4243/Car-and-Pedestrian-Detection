import cv2
#car and pedestrian video
video = cv2.VideoCapture('pedestrian video.mp4')
#video = cv2.VideoCapture('car video.mp4')

# pre-trained cars and pedestrian classifier
car_tracker_file = 'car_detector.xml'
pedestrian_tracker_file = 'pedestrian_detector.xml'

#create cars and pedestrian classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# run until car stops
while True:
    # Read the current frame
    (read_successful, frame) = video.read()

    # safe coding
    if read_successful:
        # must convert to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrian
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrian = pedestrian_tracker.detectMultiScale(grayscale_frame)

    # draw rectangles around the cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x+1,y+1), (x+w, y+h), (255, 0, 0), 2)
    
    # draw rectangles around the pedestrians 
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)


    cv2.imshow('Car and Pedestrian Detector', frame)
    key = cv2.waitKey(1)

    # stop if Q is pressed
    if key == 81 or key == 113:
        break

# release the VideoCapture object
video.release()

# end of code
print("code completed")