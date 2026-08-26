import cv2
import ctypes
import time

# ----------------------------
# Configuration
# ----------------------------

ABSENCE_TIMEOUT = 3  # seconds before lock
CAMERA_INDEX = 0
locked = False

# ----------------------------
# Face Detector
# ----------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise Exception("Failed to load face detector")

# ----------------------------
# Camera
# ----------------------------

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise Exception("Failed to open webcam")

last_seen = time.time()

print("Presence monitor started.")
print(f"PC will lock after {ABSENCE_TIMEOUT} seconds of absence.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera lost. Reconnecting...")
        cap.release()
        time.sleep(10)
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(60, 60)
    )

    if len(faces) > 0:
        last_seen = time.time()
        locked = False

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

    seconds_remaining = max(
        0,
        ABSENCE_TIMEOUT - int(time.time() - last_seen)
    )

    cv2.putText(
        frame,
        f"Lock in: {seconds_remaining}s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Presence Detection", frame)

    if (
    time.time() - last_seen >= ABSENCE_TIMEOUT
    and not locked
    ):
        print("No face detected. Locking workstation...")
        ctypes.windll.user32.LockWorkStation()
        locked = True

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()