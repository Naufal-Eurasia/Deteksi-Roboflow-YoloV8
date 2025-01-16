# Deteksi-Roboflow-YoloV8
UAS PENGOLAHAN CITRA DIGITAL
```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="spzWrqN55I8wZvPVPJcc")
project = rf.workspace("deteksi-gambar-camera-dan-laptop-yolo").project("laptop-1gzly")
version = project.version(1)
dataset = version.download("yolov8")
                

# Step 1: Install necessary libraries
!pip install ultralytics  # Install YOLOv8
!pip install matplotlib opencv-python-headless
!pip install roboflow

# Step 2: Import libraries
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files
import cv2
import numpy as np




# !pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="p9AE4cfyWZVtKr7MenmF")
# project = rf.workspace("testingws").project("object-detection-bbaki")
# version = project.version(3)
# dataset = version.download("yolov8")

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="4gEbZ28NJvB2NPSkRws0")
project = rf.workspace("deteksi-gambar-camera-dan-laptop-yolo").project("laptop-1gzly")
version = project.version(1)
dataset = version.download("yolov8")







import os

# Lihat folder tempat dataset diunduh
dataset_location = dataset.location  # dari RoboFlow download
print("Dataset downloaded to:", dataset_location)

from ultralytics import YOLO

# Buat model YOLOv8 baru
model = YOLO("yolov8n.pt")  # "yolov8n.pt" adalah versi YOLOv8 Nano

# Jalankan pelatihan dengan dataset
model.train(data="/content/Laptop---1/data.yaml", epochs=100, imgsz=100)

from ultralytics import YOLO

# Muat model terlatih
model = YOLO("/content/runs/detect/train/weights/best.pt")


dataset = version.download("yolov8")

# result = model.predict(source="/content/gol3.png", save=True, imgsz=320)

image_path = "/content/runs/detect/train/results.png"
gray_image_path = "/content/img1_gray.png"

# Baca gambar
image = cv2.imread(image_path)

# Ubah ke grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Simpan gambar grayscale
cv2.imwrite(gray_image_path, gray_image)



result = model.predict(source="/content/runs/detect/train/results.png", imgsz=640, save=True)


# Ambil elemen pertama dari hasil prediksi
image_result = result[0]

# Ekstrak bounding box, confidence, dan class
for box in image_result.boxes.data:
    # box adalah tensor [x_min, y_min, x_max, y_max, confidence, class]
    x_min, y_min, x_max, y_max, confidence, class_id = box.tolist()

    # Cetak bounding box
    print(f"Bounding Box: ({x_min:.2f}, {y_min:.2f}, {x_max:.2f}, {y_max:.2f})")
    print(f"Confidence: {confidence:.2f}")
    print(f"Class ID: {int(class_id)}")

# Menghitung jumlah bounding box
total_bounding_boxes = len(image_result.boxes.data)
print(f"Total Bounding Boxes: {total_bounding_boxes}")

# Menampilkan hasil deteksi
from IPython.display import Image, display
image_path_with_predictions = image_result.plot()  # Mengembalikan array gambar

# Tampilkan menggunakan Matplotlib
import matplotlib.pyplot as plt
plt.imshow(image_path_with_predictions)
plt.axis("off")
plt.show()

import cv2

# Buka gambar asli
img = cv2.imread("/content/Laptop---1/test/images/18ef641954373dc3_jpg.rf.fdfe3f2c6ae78b11369f0c43582c220b.jpg")

# Loop untuk menggambar bounding box
for box in image_result.boxes.data:
    x_min, y_min, x_max, y_max, confidence, class_id = box.tolist()
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    # Gambar bounding box
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    label = f"Class: {int(class_id)}, Conf: {confidence:.2f}"
    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Tampilkan gambar dengan Matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()


# Step 3: Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]



# Step 4: Load YOLO model
model = YOLO('yolov8n.pt')  # Use a pre-trained YOLOv8 model (nano version for speed)

# Step 5: Perform object detection
results = model(image_path)  # Run inference on the uploaded image

# Step 6: Visualize results
# Save the annotated image
annotated_img = results[0].plot()  # Create an annotated image (numpy array)

cv2.imwrite("runs/detect/predictions.jpg", annotated_img)

# Display the annotated image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("YOLO Detected Objects")
plt.show()
```
Hasil gambar 

![image](https://github.com/user-attachments/assets/1494d9e7-2d78-4a58-aa6c-d09464a0ab1c)
![image](https://github.com/user-attachments/assets/3158fe37-63c2-44b8-a154-c2b826553a33)
![image](https://github.com/user-attachments/assets/d3543249-71d1-48a6-be7f-2ee7610362a7)

Link tautan google colab
https://colab.research.google.com/drive/1hTvr4JhJKCcAY9wFKfFLt9aCP4luaQj2


