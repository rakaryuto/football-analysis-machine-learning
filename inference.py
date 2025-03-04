from ultralytics import YOLO

model = YOLO('./models/best.pt')

result_frames = model.predict('./input-videos/08fd33_4.mp4', save=True)
print(result_frames[0])
print('============================')

for box in result_frames[0].boxes : 
    print(box)