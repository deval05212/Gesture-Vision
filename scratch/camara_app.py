import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    
    # Black and white filter
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Thermal
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # output = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # colors
    # output = cv2.bitwise_not(frame)
    # Increase brightness

    cv2.imshow("Beauty Filter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()