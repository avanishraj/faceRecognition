import cv2, numpy, os, urllib, imutils, pyttsx3

datasets = "dataset"
engine = pyttsx3.init()
redLower = (100, 50, 50)
redUpper = (187, 100, 100)
print("Training...")
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectPath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectPath):
            path = subjectPath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        # print(labels)
        id += 1
(images, labels) = [numpy.array(lis) for lis in [images, labels]]
# print(images, labels)
(width, height) = (130, 100)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)
url = "http://192.168.1.3:8080/shot.jpg"
haar_file = "haarcascade_frontalface_default.xml"
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(haar_file)
cnt = 0
while True:
    _, img = cam.read()
    # imagePath = urllib.request.urlopen(url)
    # imageNp = numpy.array(bytearray(imagePath.read()), dtype=numpy.uint8)
    # img = cv2.imdecode(imageNp, -1)
    img = imutils.resize(img, width = 450)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussImg = cv2.GaussianBlur(img, (11,11), 0)
    hsvImg = cv2.cvtColor(gaussImg, cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSVImg", hsvImg)
    mask = cv2.inRange(hsvImg, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        if radius > 10:
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2) 
            cv2.circle(img, center, 5, (0, 255, 255), -1)
            if radius > 250:
                print("Stop")
            else:
                if(center[0] < 150):
                    engine.say("on your left")
                elif(center[0] > 450):
                    engine.say("on your right")
                elif(center[0] < 250):
                    engine.say("center")
                else:
                    engine.say("stop")
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 5)  #detectMultiScale method of the face object to detect faces in the image.
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)
        faceOnly = grayImg[ y:y+h, x:x+w]
        resizeImg = cv2.resize(faceOnly, (width, height))
        prediction = model.predict(resizeImg)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
        if prediction[1] < 1500:
            name = names[prediction[0]]
            confidence = prediction[1]
            cv2.putText(img, "%s - %.0f" %(names[prediction[0]], prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            print(names[prediction[0]])
            cnt = 0
            engine.say("This is" + name)
            engine.runAndWait()
        else :
            cnt += 1
            cv2.putText(img, "unknown" %(names[prediction[0]], prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            if(cnt > 80):
                print("Unkown person")
                cv2.imwrite("unknownPerson.jpg", img)
                cnt = 0
    cv2.imshow("FaceRecognition", img)
    if ord('q') == cv2.waitKey(10):
        break
cam.release()
cv2.destroyAllWindows()