## ROS

## Table of Content
- [Dira's Package](#Dira)
- [Libraries's requirements](#Libraries)
- [About ROS](#About-ROS)
- [Semantic Segmentation](#Semantic-Segmentation)
- [The Navigation Stack](#the-navigation-stack)
	* [RTABMap](#RTABMap)
	* [Path Planning](#Path-Planning)
	* [Vehicle Motion Control](#Vehicle-Motion-Control)

## Dira

Ban tổ chức có đưa 3 package cơ bản để có thể khởi động xe: 

1. [dira_mpu9250_controller](https://github.com/datvuthanh/Digital-Race/tree/master/src/dira_mpu9250_controller)
2. [dira_pca8266_controller](https://github.com/datvuthanh/Digital-Race/tree/master/src/dira_pca8266_controller)
3. [dira_peripheral_controller](https://github.com/datvuthanh/Digital-Race/tree/master/src/dira_peripheral_controller)

Hardware requirements:
1. Jetson TX2
2. LCD 16x4
3. Arduino Mini
4. Servo SG90
5. Orrbec Astra Camera
6. RPLidar A2
7. Infrared sensor
 
<center>
<img src="../images/car_1.png" alt="image" width="640"/>
</center>

## Libraries

Các thư viện tối thiểu cần phải cài trên mạch jetson-tx2:

1. Thư viện Astra camera tích hợp ROS (http://wiki.ros.org/astra_camera)
2. Thư viện rplidar_ros cho LiDAR 2D - rplidar A2 (http://wiki.ros.org/rplidar)
3. Thư viện rtabmap và navigation (http://wiki.ros.org/rtabmap_ros và http://wiki.ros.org/navigation)

Ngoài ra, để đạt hiệu năng tốt nhất trên jetson-tx2 chúng tôi có cài đặt thêm một số thư viện khác:

1. Thư viện numba (https://github.com/jefflgaol/Install-Packages-Jetson-ARM-Family)
2. Thư viện Tensorflow, Keras
3. Tăng tốc độ khởi động TensorRT trên jetson-tx2 (https://jkjung-avt.github.io/tf-trt-revisited)