import cv2 as cv

pedestrain_tracker = cv.CascadeClassifier("haarcascade_fullbody.xml")
cars_tracker = cv.CascadeClassifier("cars.xml")
# img = cv.imread("image1.png")
# cars_img  = cv.imread("car1.png")
webcam = cv.VideoCapture('video.mp4')


while True:
    successful_frame_read, frame = webcam.read()

    grayed_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    body_coordinates = pedestrain_tracker.detectMultiScale(grayed_img)
    car_coordinates = cars_tracker.detectMultiScale(grayed_img) 
    # print(body_coordinates)

    # body1 = body_coordinates[0]
    # # print(body1)
    # (x,y,w,h) = body1
    # cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)

    for (x, y, w, h) in body_coordinates:
        cv.rectangle(frame,(x,y), (x+w,y+h), (0,0,255), 2)
    
    for (x, y, w, h) in car_coordinates:
        cv.rectangle(frame,(x,y), (x+w,y+h), (0,255,255), 2)


    cv.imshow("Pallav Pedestrain Detector App", frame)
    key = cv.waitKey(1)

    # stop if q is pressed

    if key == 81 or key == 113:
        exit()

print("Code completed")
