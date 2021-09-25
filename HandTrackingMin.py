import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    # go through all the hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # get landmark information, id numbers
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                # find the position
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

                # if landmark id is 0 then draw a purple circle
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)

            # draw hands
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate framerate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)