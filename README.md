<center>
<img src="./images/cover.png" alt="Cover"/>
</center>

### Demo:

<center>
<img src="./images/demo.gif" width="242" height="137"/>
</center>

Welcome! This is an open-source self-driving car aimed for rapid prototyping, deep learning, and robotics research. The system currently runs on a jetson tx2 module. Here are our goals:

### Goals:
Research and develop a deep learning-driven self-driving car. The vehicle should be able to finish the race. 

### Role 
To know the role, please read [documentation](./role/README.md). 

#### The modules in this project.

1. Semantic Segmentation
2. Object Detection
3. Mapping with rtabmap
4. Path planning with ROS nav stack. 
5. Driveless System

For the full documentation of the development process, please visit my website: [datvuthanh.github.io](https://datvuthanh.github.io)

## Table of Content
- [Introduction](#Introduction)
- [Try it out](#Try-it-out)
- [About ROS](#About-ROS)
- [Semantic Segmentation](#Semantic-Segmentation)
- [The Navigation Stack](#the-navigation-stack)
	* [RTABMap](#RTABMap)
	* [Path Planning](#Path-Planning)
	* [Vehicle Motion Control](#Vehicle-Motion-Control)

## Introduction
to do it

## Try it out
Before you jump in, let me describe the hardware requirement for this project. **A webcam is the minimum requirment.** At this point, you can only run the whole system on the actual self-driving vehicle. ROS will throw warnings (even errors) at you if you don't have the hardware connected to your Linux machine. **If you don't have access to the hardware setup, don't worry** 👇
 
- The best way is to download and play back the ROS bags. [coming soon...]
- You can tryout individual packages and nodes, and might find them helpful for your own projects. 
- You can also tryout the CARLA simulator. (Maybe even improve the current system.)

To compile the project:

##### Requirements

1. Make sure that you have [ROS](http://wiki.ros.org/melodic/Installation/Ubuntu) installed on your computer. (I am using ROS Melodic)
2. Make sure you have all the [dependencies](./src/README.md) installed. 

##### Clone & Compile

1. Clone the repository. `$ git clone https://github.com/datvuthanh/Digital-Race.git`
2. `$ cd Digital-Race` 
3. `$ cp -r src/. ~/catkin_ws/src/.`
4. `$ cd ~/catkin_ws/`
5. `$ catkin_make`
6. `$ source devel/setup.bash`

## About ROS
This project uses ROS. __For more information on ROS, nodes, topics and others please refer to the ROS [README](./src/README.md).__

## Semantic Segmentation
The cart understands its surrounding  through semantic segmentation, which is a technique in computer that classifies each pixel in an image into different categories. The vehicle can also make decisions based on the segmentic segmentation results. The cart can change its speed based on the proximity to nearby obstacles.

<center>
<img src="./images/pspnet.png" alt="Drawing" width="640"/>
</center>

We deployed the PSPNet architecture for segmentation. PSPNet is design to work well in realtime applications. For more information, please visit the [paper](https://arxiv.org/pdf/1612.01105.pdf). We collect dataset for training and the python code for training and inferencing are located in the `segmentation` directory.

[VIDEO DEMO](https://youtu.be/RMJ9s7XbxDs)

<a name="the-navigation-stack" > </a>

## The Navigation Stack

![](./images/nav_stack.png)

The self-driving vehicle uses a modified version of the ROS navigation stack. The flowchart above illustrate the mapping and path planning process. First, I create a detailed map of the environment with `rtabmap_ros`. With that global map, I use the localization feature of `rtabmap_ros` and the odom feature of the zed camera system to localize and plan paths. 

<a name="RTABMap" > </a>

### RTABMap

`rtabmap` (realtime appearance based mapping) allows me to construct a global map of the environment. For more information on the mapping package, please check out this [`.launch` file](./ros/src/navigation/mapping/launch/rtab_mapping.launch). 

<center>
	<img src="./images/rtab-map.png" alt="Drawing" width="640"/>
</center> 

<a name="Path%20Planning" > </a>

### Path Planning

The project uses the [`move_base`](http://wiki.ros.org/move_base) node from the navigation stack. The image below shows the costmap (in blue and purple), and the global occupancy grid (in black and gray). `move_base` also plans the local and global path. Global paths are shown in green and yellow below. You can find the `yaml` files [here](./ros/src/navigation/path_planning/params). 

<center>
	<img src="./images/path_plan_1.png" alt="Drawing" width="640"/>
</center>

<a name="Vehicle%20Motion%20Control" > </a>

### Vehicle Motion Control

The move base node publishes `/cmd_vel` commands, which are processed and sent directly to the vehicle. 

<center>
	<img src="./images/vehicle_side.png" alt="Drawing" width="640"/>
</center>

# Contact / Info
If you are interested in the detailed development process of this project, you can visit Neil's blog at [neilnie.com](https://neilnie.com) to find out more about it. Neil will make sure to keep you posted about all of the latest development on the club.

**Contributors:**

**Dat Vu (Leader)** | [Email](mailto:stephen.t.vu@hotmail.com) | [Github](https://www.github.com/datvuthanh) | [Website](https://datvuthanh.github.io/)

<img src="./images/datvu.jpg" alt="Drawing" width="80"/>

**Huy Phan** | [Email](mailto:HuyPQHE141762@fpt.edu.vn) 

<img src="./images/huyphan.png" alt="Drawing" width="80"/>

**Tra Dinh** | [Email](mailto:trandhe140661@fpt.edu.vn) 

<img src="./images/tradinh.png" alt="Drawing" width="80"/>

**Hai Anh Tran** | [Email](mailto:anhthhe141545@fpt.edu.vn) | [Github](https://github.com/AnhTH-FUHN)

<img src="./images/haianh.jpg" alt="Drawing" width="80"/>

