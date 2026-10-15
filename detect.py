import threading
import time
from ultralytics import YOLO
import cv2
import pyttsx3


engine = pyttsx3.init()
engine.setProperty("rate", 175)  
engine.setProperty("volume", 1.0)  


def speak_async(text):
  """Runs text-to-speech in the background so the video doesn't freeze"""

  def run():
    engine.say(text)
    engine.runAndWait()

  threading.Thread(target=run, daemon=True).start()



model = YOLO("yolo11l.pt")
cap = cv2.VideoCapture(0)


previous_objects = set()
last_spoken_time = 0
cooldown_seconds = 3  

print(
    "Starting real-time object detection, counting, and voice announcements..."
)

while cap.isOpened():
  success, frame = cap.read()

  if not success:
    print("Camera not detected!")
    break

 
  results = model(frame)
  annotated_frame = results[0].plot()

 
  boxes = results[0].boxes
  total_objects = len(boxes)


  object_counts = {}
  for box in boxes:
    cls_id = int(box.cls[0])
    name = model.names[cls_id]
    object_counts[name] = object_counts.get(name, 0) + 1


  
  current_objects_set = frozenset(object_counts.items())



  y_offset = 40
  cv2.putText(
      annotated_frame,
      f"Total Objects: {total_objects}",
      (20, y_offset),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (0, 255, 0),
      2,
  )

  for obj_name, count in object_counts.items():
    y_offset += 30
    text = f"- {obj_name}: {count}"
    cv2.putText(
        annotated_frame,
        text,
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )


  
  current_time = time.time()
  if current_objects_set != previous_objects and total_objects > 0:
    if current_time - last_spoken_time > cooldown_seconds:
      # Build dynamic voice message (e.g., "1 person and 1 bottle detected")
      desc_list = [f"{count} {name}" for name, count in object_counts.items()]
      message = f"Detected {total_objects} objects: " + ", ".join(desc_list)

      speak_async(message)
      previous_objects = current_objects_set
      last_spoken_time = current_time

 
  elif total_objects == 0:
    previous_objects = set()

  cv2.imshow("Dynamic Object Detection & Counter", annotated_frame)


  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()