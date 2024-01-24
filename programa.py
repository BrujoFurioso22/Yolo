from ultralytics import YOLO

model = YOLO('https://github.com/BrujoFurioso22/Yolo/blob/main/best.pt')

results = model('https://github.com/BrujoFurioso22/Yolo/blob/main/imagenes/c1.png')

print(results)