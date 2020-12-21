
## Table of Content

- [Labeling](#Labeling)

- [Training](#Training)

- [Architecture](#Architecture)

- [Result](#Architecture)

- [How it works](#How-it-works)

### Labeling
Để label dữ liệu, chúng tôi sử dụng công cụ có tên là labelme. 

Công cụ này lưu file đã label dạng .json nhưng chúng ta cũng có thể sử dụng một thư viện nằm trong công cụ cho phép biến đổi các file .json thành file segment dạng ảnh (.jpg, .png).

### Training


Ý tưởng cốt lõi của chúng tôi: Từ một ảnh đầu vào có thể sử dụng deep learning/semantic segmentation để segment thành các object khác nhau như: **Line**, **Road**, **Background**, **Traffic Signs**. 

Package này sẽ nêu rõ các vấn đề tại sao chúng tôi lại sử dụng các mô hình và metrics. Việc sử dụng cho mục đích gì sẽ được nêu rõ ở một package khác trong ROS có tên là **fptu_architecture** nơi thực hiện các thuật toán lái xe và dự đoán góc lái.

### Architecture

Về kiến trúc, chúng tôi sử dụng kiến trục mạng PSPNet với backbone efficientnetb4 được train trên tập imagenet. Kích thước ảnh đầu vào là (144,144) với 4 classes: Line, Road, Traffic Signs, Background.

<center>
<img src="../images/pspnet.png" />
</center>
Về loss, chúng tôi kết hợp hai hàm loss: focal loss và weighted dice loss với trọng số của mỗi class được tính bằng tổng pixel của mỗi class chia cho tổng pixel của toàn bộ tập ảnh huấn luyện. 

Ngoài ra, chúng tôi có kết hợp một số hàm loss khác với nhau như jaccard loss + focal loss (hoặc weighted cross-entropy) kết quả huấn luyện không đạt hiệu quả bằng focal loss và weighted dice loss. 

<center>
<img src="../images/focal.png" />
</center>

<center>
<img src="../images/fb_score.png" />
</center>

Với các tham số cho dice loss, chúng tôi muốn chú trọng nhiều hơn true positive nên chúng tôi đã đặt hệ số beta = 0.5 bởi vì với sa hình thi đấu, việc segment thiếu road sẽ an toàn hơn việc segment thừa road. 

## Result

Kết quả đạt được: IoU 88%, F1-Score 94%, F1-Score Road: 98%, F1-Score Background 96%, F1-Score Line 88%.

## How it works

Dataset: https://drive.google.com/file/d/1EKA7VGdKjevT855ycYW0BOaI8u0r2Sa_/view?usp=sharing

Dự án được chúng tôi training trên GPU Titan với 20 phút training.Thực tế lúc đi thi chúng tôi không cần dán thêm bất kì data nào nhưng để đảm bảo mọi thứ hoạt động tốt nên đã label thêm 500 data.
