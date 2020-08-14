### Labeling
Để label dữ liệu, chúng tôi sử dụng công cụ có tên là labelme. 

Công cụ này lưu file đã label dạng .json nhưng chúng ta cũng có thể sử dụng một thư viện nằm trong công cụ cho phép biến đổi các file .json thành file segment dạng ảnh (.jpg, .png).

### Training

<center>
<img src="../images/segment.gif" width="768" height="384"/>
</center>

Ý tưởng cốt lõi của chúng tôi: Từ một ảnh đầu vào có thể sử dụng deep learning/semantic segmentation để segment thành các object khác nhau như: **Line**, **Road**, **Background**, **Traffic Signs**. 

Package này sẽ nêu rõ các vấn đề tại sao chúng tôi lại sử dụng các mô hình và metrics. Việc sử dụng cho mục đích gì sẽ được nêu rõ ở một package khác trong ROS có tên là **fpt_architecture** nơi thực hiện các thuật toán lái xe và dự đoán góc lái.
