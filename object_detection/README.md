

## Table of Content

- [Object Detection](#Object-Detection)

- [Preprocessing](#Preprocessing)

- [Training](#Training)

- [Architecture](#Architecture)

- [How it works](#How-it-works)

### Object Detection

Trong cuộc thi, sẽ có 6 loại biển báo. Khi xe tự lái gặp các biển báo sẽ thực hiện các hành động khác nhau. 
Vì vậy nhiệm vụ của package này là làm thế nào để có thể nhận biết và phân biệt các loại biển báo mà vẫn đảm bảo tốc độ xử lý theo thời gian thực trên mạch jetson-tx2. 

### Preprocessing


### Training 
SSD7 Method 

### Architecture


### Results

<center>
<img src="./images/SSD_1.png" alt="Cover"/>
</center>

<center>
<img src="./images/SSD_2.png" alt="Cover"/>
</center>

<center>
<img src="./images/SSD_3.png" alt="Cover"/>
</center>

### Dataset
[Dataset](https://drive.google.com/file/d/1NGrKWHc1z_4bOh2huWHC8kZsUZFXOku-/view?usp=sharing)
### How it works 

1. Train model ```python3 train.py```

2. Để predict object trong file ```predict.py``` sửa đường dẫn weight và đường dẫn ảnh. Sau đó chạy bằng lệnh ```python3 predict.py``` 

3. Để convert model về dạng frozen graph (.pb) sửa đường dẫn weight và chạy bằng lệnh ```python3 convert.py```

   
