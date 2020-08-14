#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

void depthImageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    ros::Time begin = ros::Time::now();

    cv_bridge::CvImagePtr cv_ptr;
    //ROS_INFO("%s",msg->encoding);
    if (msg->encoding == "16UC1"){
						sensor_msgs::Image img;
						img.header = msg->header;
						img.height = msg->height;
						img.width = msg->width;
						//ROS_INFO("WIDTH %d",img.width);
						img.is_bigendian = msg->is_bigendian;
						img.step = msg->step;
						img.data = msg->data;
						img.encoding = "mono16";

						cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
						cv::imshow("Depth",cv_ptr->image);
    }
    /*    
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::imshow("Depth", cv_ptr->image);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    */
    ros::Duration now = ros::Time::now() - begin;
    
    ROS_INFO("TIME for Depth Processing: %f",now);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    ros::WallTime start_, end_;
    
    start_ = ros::WallTime::now();

    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::imshow("RGB", cv_ptr->image); 
        cv::waitKey(1);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }

    end_ = ros::WallTime::now();

    // print results
    double execution_time = (end_ - start_).toNSec(); //* 1e-6;
    
    ROS_INFO_STREAM("Exectution time (ms): " << execution_time);  
}

int main(int argc, char **argv)
{
    
    ros::init(argc, argv, "video_stream");
    ros::NodeHandle nh, nh2;
    

    cv::namedWindow("RGB");
    cv::namedWindow("Depth");
    
    // cv::startWindowThread();
    while(ros::ok()){

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/camera/rgb/image_raw", 1, imageCallback);

    image_transport::ImageTransport it2(nh2);
    image_transport::Subscriber sub2 = it2.subscribe("/camera/depth/image_raw", 1, depthImageCallback);
    
    ros::spin();

    }

    cv::destroyWindow("RGB");

    cv::destroyWindow("Depth");
}
