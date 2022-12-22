import imutils
import cv2 as cv
import mediapipe as mp
import functions as fun

# Import cascade file for facial recognition
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
# Enable camera
cap = cv.VideoCapture(0)

while True:
    # Check if the video stream is open
    if not cap.isOpened():
        print('Error: Could not open the video stream')

    # Read a frame from the video stream
    _, frame = cap.read()

    # Object detection
    mask = fun.make_mask_for_image(frame, lower=[170, 50, 50], upper=[180, 225, 225])
    # A list of contours and (hierarchy of contours), where each contour is a list of points outlining the boundary of an object
    contour = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # The grab_contours function extracts the list of contours and returns it as a simple list.
    contour = imutils.grab_contours(contour)

    #  iterates over these contours to draw rectangles around objects that meet the size criteria
    for c in contour:
        rect = cv.boundingRect(c)
        # skip if the width (rect[2]) or the height (rect[3]) of the enclosing rectangle are smaller than 100 pixels.
        if rect[2] < 100 or rect[3] < 100:
            continue
        x, y, w, h = rect
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print("Red Object is detected")

    # Hand detection
    # Hand landmarks and handedness of each detected hand
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    # Drawing different types of annotations on video frames.
    mpDraw = mp.solutions.drawing_utils
    # Fingertips (The ids)
    tipIds = [4, 8, 12, 16, 20]
    # Flip the image vertically (around the x-axis).
    frame = cv.flip(frame, 1)
    # Convert frame from BGR to RGB
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Processes an RGB frame -> hand landmarks
    results = hands.process(frameRGB)
    # List of [id, center x, center y]
    lmList = []
    # List of fife fingers [f1, f2, f3, f4, f5], f can either contains 0, 1
    fingersRightHand = []
    fingersLeftHand = []

    # A hand is detected
    if results.multi_hand_landmarks:
        #  Iterates over the multi_hand_landmarks attribute
        for handLms in results.multi_hand_landmarks:
            # iterates over the landmarks, which is a list of Landmark objects representing the points on the hand.
            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape  # width, height, center
                # calculates the center of each landmark by multiplying the x and y coordinates of the landmark object by the width and height of the frame,
                cx, cy = int(lm.x * w), int(lm.y * h)  # center x, center y
                lmList.append([id, cx, cy])
                # Draws the landmarks and the connections on the image
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

                # Mark index finger with green circle
                if id == 8:
                    cv.circle(frame, (cx, cy), 20, (0, 255, 0), cv.FILLED)

                # The list must contains 21 (0 ... 20) elements
                if len(lmList) == 21:

                    handNumber = fun.get_detected_hand(results)

                    # Checks whether the "right" hand or both hands are detected, and uses the landmarks in lmList to determine whether each finger is raised or not
                    if handNumber == 1 or handNumber == 0:
                        print("Right Hand")
                        # special case for the thumb "right hand"
                        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 2][1]:
                            fingersRightHand.append(1)
                        else:
                            fingersRightHand.append(0)
                        for tip in range(1, 5):
                            if lmList[tipIds[tip]][2] < lmList[tipIds[tip] - 2][2]:
                                fingersRightHand.append(1)
                            else:
                                fingersRightHand.append(0)

                    # Checks whether the "left" hand or both hands are detected, and uses the landmarks in lmList to determine whether each finger is raised or not
                    if handNumber == 2 or handNumber == 0:
                        print("Left Hand")
                        # special case for the thumb "left hand"
                        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 2][1]:
                            fingersLeftHand.append(1)
                        else:
                            fingersLeftHand.append(0)
                        for tip in range(1, 5):
                            if lmList[tipIds[tip]][2] < lmList[tipIds[tip] - 2][2]:
                                fingersLeftHand.append(1)
                            else:
                                fingersLeftHand.append(0)
                    # counts the number of raised fingers on both hands by counting the number of 1s in the fingersRightHand and fingersLeftHand lists
                    totalFingers = fingersRightHand.count(1) + fingersLeftHand.count(1)
                    cv.putText(frame, f'{totalFingers}', (40, 80), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

    # Face detection
    #  Convert the input frame from the BGR color space to grayscale
    imgGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Getting corners around the face (x, y, w, h) x,y => upper left corner of the rectangle; w,h => specify the width and height of the rectangle
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor

    # drawing bounding box around face
    for (x, y, w, h) in faces:
        img = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the flipped frame
    cv.imshow('Exam requirements', frame)

    # Check if the user pressed the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream
cap.release()
cv.destroyAllWindows()
