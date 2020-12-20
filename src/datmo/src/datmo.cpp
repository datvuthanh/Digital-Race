/*
 * Copyright (c) 2020, Robobrain.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/* Author: Konstantinos Konstantinidis */

#include "datmo.hpp"
#include <std_msgs/Int8.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <math.h>
#include <time.h>

#pragma omp parallel for
static inline double calEuclidDistance(double x, double y){
  return sqrt(x*x + y*y);
}

static inline void wait(int seconds)
{
	clock_t ew;
	ew = clock() + seconds * CLOCKS_PER_SEC;
	ROS_INFO("WAIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
	while (clock() < ew) {
    	 //ROS_INFO("DM CUOC DOI!!!!!!!!!!!!!!!!!");
  }
}

static inline int findClusterNearest(vector<Cluster> clusters){
  double min=999999;
  double min_i = 0;
  for (unsigned int i =0; i<clusters.size();i++){

    // track_array_box_kf.tracks.push_back(clusters[i].msg_track_box_kf);

    //TRADINH ADD TO FIND NEAREST CLUSTER  (190-195)
    double cal_i = sqrt(clusters[i].meanX()*clusters[i].meanX()+clusters[i].meanY()*clusters[i].meanY());
    if(cal_i<min){
      min=cal_i;
      min_i=i;
    }
  }
  return min_i;
}
static inline int findClusterNearestWithCondition(vector<Cluster> clusters){
  double min=999999;
  double min_i = -1;
  for (unsigned int i =0; i<clusters.size();i++){

    // track_array_box_kf.tracks.push_back(clusters[i].msg_track_box_kf);

    //TRADINH ADD TO FIND NEAREST CLUSTER  (190-195)
    double cal_i = sqrt(clusters[i].meanX()*clusters[i].meanX()+clusters[i].meanY()*clusters[i].meanY());
    if(cal_i<min && clusters[i].meanY()>0 && clusters[i].pointsize > 4){
      min=cal_i;
      min_i=i;
    }
  }
  return min_i;
}
// static inline bool checkObjectBeforeCar(vector<Cluster> clusters){
//   //if have return id 
//   //if no return -1
  
//  for(int i =0;i< clusters.size();i++){

//       if(clusters[i].pointsize > 5 && clusters[i].meanX()>-2 && clusters[i].meanX()<-0.3 && clusters[i].meanY()>-0.2 && clusters[i].meanY()<2){
//         // marker_array.markers.push_back(clusters[i].getBoundingBoxCenterVisualisationMessage());
//         return true;
//       }
//     }
//   return false;
// }

static inline int findObjectBeforeCar(vector<Cluster> clusters,int min_pointsize_obstacle,double min_height_car_region,double max_height_car_region,double min_width_car_region, double max_width_car_region){
  //return id of cluster
  //return -1 if dont have cluster
   for(int i =0;i< clusters.size();i++){

      if(((clusters[i].length_box >0.25 && clusters[i].length_box <0.37)|| (clusters[i].width_box > 0.25 && clusters[i].width_box<0.37)) &&
       clusters[i].pointsize > min_pointsize_obstacle && 
       clusters[i].closest_corner_point.first > -max_height_car_region && 
       clusters[i].closest_corner_point.first < -min_height_car_region && 
       clusters[i].closest_corner_point.second > min_width_car_region && 
       clusters[i].closest_corner_point.second < max_width_car_region){
        
        return clusters[i].id;
      }
    }
  return -1;
}

static inline int findObjectBeforeCar3(vector<Cluster> clusters,int min_pointsize_obstacle,double min_height_car_region,double max_height_car_region,double min_width_car_region, double max_width_car_region){
  //return id of cluster
  //return -1 if dont have cluster
   for(int i =0;i< clusters.size();i++){
      double x = clusters[i].closest_corner_point.first;
      double y = clusters[i].closest_corner_point.second;
      if((clusters[i].length_box >0.20 && clusters[i].length_box <0.37)|| (clusters[i].width_box > 0.25 && clusters[i].width_box<0.37) || 
	(clusters[i].length_box > 0.5 && clusters[i].length_box < 0.7) || (clusters[i].width_box >0.5 && clusters[i].width_box <0.7)){
        if(clusters[i].pointsize > min_pointsize_obstacle){
          ROS_INFO("arctan: %f",atan(-x/(y-0.2)));
          if((x > -3 && x < 0 && y >= -0.2 && y <= 0.2) || 
            (calEuclidDistance(x,y-0.2) < 3 && atan(-x/(y-0.2)) > PI/6 && atan(-x/(y-0.2)) < PI/2 && x < 0 )){
            // || (x > -1.5 && x < 0 && y > 0.2 && y < 2.5)){
                ROS_INFO("KHOANG CACH TUYET DOI : %f",calEuclidDistance(x,y));
                ROS_INFO("GOC LECH : %f",atan(-x/y));
                return clusters[i].id;
            }
        } 
        }
      }
  return -1;
}

static inline int findObjectBeforeCar4(vector<Cluster> clusters,int min_pointsize_obstacle,double min_height_car_region,double max_height_car_region,double min_width_car_region, double max_width_car_region){
  //return id of cluster
  //return -1 if dont have cluster
   for(int i =0;i< clusters.size();i++){
      double x = clusters[i].meanX();
      double y = clusters[i].meanY();
      if((clusters[i].length_box >0.25 && clusters[i].length_box <0.37)|| (clusters[i].width_box > 0.25 && clusters[i].width_box<0.37)){
        if(clusters[i].pointsize > min_pointsize_obstacle){
          ROS_INFO("arctan: %f",atan(-x/(y-0.2)));
          if((x > -3.0 && x < -0.3 && y >= -0.2 && y <= 0.2) || 
            (calEuclidDistance(x,y-0.2) < 3.0 && atan(-x/(y-0.2)) > PI/4 && atan(-x/(y-0.2)) < PI/2 )){
                return clusters[i].id;
            }
        } 
        }
      }
  return -1;
}


static inline int findObjectBeforeCar1(vector<Cluster> clusters,int min_pointsize_obstacle,double min_height_car_region,double max_height_car_region,double min_width_car_region, double max_width_car_region){
  //return id of cluster
  //return -1 if dont have cluster
   for(int i =0;i< clusters.size();i++){

      if(((clusters[i].length_box >0.25 && clusters[i].length_box <0.37)|| (clusters[i].width_box > 0.25 && clusters[i].width_box<0.37)) &&
       clusters[i].pointsize > min_pointsize_obstacle && 
       clusters[i].meanX() > -max_height_car_region && 
       clusters[i].meanX() < -min_height_car_region && 
       clusters[i].meanY() > min_width_car_region && 
       clusters[i].meanY() < max_width_car_region){
        
        return clusters[i].id;
      }
    }
  return -1;
}
static inline bool checkHaveObjRightCar(vector<Cluster> clusters){
     for(int i =0;i< clusters.size();i++){

      if(((clusters[i].length_box >0.25 && clusters[i].length_box <0.37)|| (clusters[i].width_box > 0.25 && clusters[i].width_box<0.37)) && clusters[i].pointsize > 5 && clusters[i].meanX()>-2 && clusters[i].meanX()< 1 && clusters[i].meanY() > 0.2 && clusters[i].meanY()<0.6){
        
        return true;
      }
    }
  return false;
}

// static inline int trackkingObject(vector<Cluster> clusters){
//   //if have return id 
//   //if no return -1
//   int max_pointsize = 0;
//   int cluster_id = -1;
//   for(int i =0;i < clusters.size();i++){
//     if(clusters[i].pointsize > 6 && clusters[i].pointsize >max_pointsize  && clusters[i].meanX()>-4.2 && clusters[i].meanX() < 0 && clusters[i].meanY() < 7 && clusters[i].meanY() > 0){
//       max_pointsize = clusters[i].pointsize;
//       cluster_id = clusters[i].id;
//     }
//   }
//   return cluster_id;
// }

#pragma omp parallel for
static inline int checkSign(vector<Cluster> clusters,int max_pointsize_sign, double min_width_sign_region, double max_width_sign_region, double min_height_sign_region, double max_height_sign_region){
  for(int i =0;i<clusters.size(); i++){
    if(clusters[i].pointsize < max_pointsize_sign && clusters[i].meanY() > min_width_sign_region && clusters[i].meanY()< max_width_sign_region && clusters[i].meanX()>-max_height_sign_region && clusters[i].meanX()< -min_height_sign_region ){
      return 1;
    }

  }
  return 0; 
}


// DAT VU ADD FOR SSD DETECTION
// #pragma omp parallel for
// static inline Cluster TrackingSign_SSD(vector<Cluster> clusters,int max_pointsize_sign, double min_width_sign_region, double max_width_sign_region, double min_height_sign_region, double max_height_sign_region){
//   for(int i =0;i<clusters.size(); i++){
//     if(clusters[i].pointsize < max_pointsize_sign && clusters[i].meanY() > min_width_sign_region && clusters[i].meanY()< max_width_sign_region && clusters[i].meanX()>-max_height_sign_region && clusters[i].meanX()< -min_height_sign_region ){
//       return clusters.at(i);
//     }
//   }
// }

#pragma omp parallel for
static inline bool checkRightCar(vector<Cluster> clusters){
    for(int i =0;i<clusters.size(); i++){
      if(clusters[i].meanY() > 0.2 && clusters[i].meanY()< 0.8 && clusters[i].meanX()> -2 && clusters[i].meanX()< 0.7 ){
        return true;
      }

  }
  return false;
}






static inline visualization_msgs::Marker getCenter(geometry_msgs::Point p) {

    visualization_msgs::Marker boxcenter_marker;
    boxcenter_marker.type = visualization_msgs::Marker::POINTS;
    boxcenter_marker.header.frame_id = "laser";
    boxcenter_marker.header.stamp = ros::Time::now();
    boxcenter_marker.ns = "bounding_box_center";
    boxcenter_marker.action = visualization_msgs::Marker::ADD;
    boxcenter_marker.pose.orientation.w = 1.0;    
    boxcenter_marker.scale.x = 0.1;
    boxcenter_marker.scale.y = 0.1;  
    boxcenter_marker.color.a = 1.0;
    boxcenter_marker.color.r = 1;
    boxcenter_marker.color.g = 1;
    boxcenter_marker.color.b = 0;
    boxcenter_marker.id = 1000;

    boxcenter_marker.points.push_back(p);

  return boxcenter_marker;
}
// static inline int decisiontree(cluster cluster){
//   double cluster.
// }
int count_sign_clt = 0;
bool is_obj_detect = false;
// bool is_avoid_object = false;
void callbacksign(const std_msgs::Int8::ConstPtr& msg){
  count_sign_clt = msg->data;
  // ROS_INFO("I heard: [%d]", msg->data);
}
void callbackobj(const std_msgs::Bool::ConstPtr& msg_avd){
  is_obj_detect = msg_avd->data;
  ROS_INFO("I heard: [%d]", msg_avd->data);
}

bool bt4_status = false;

void btnCallback(const std_msgs::Bool::ConstPtr& msg_avd){
  bt4_status = msg_avd->data;
  if (bt4_status == true){
  count_sign_clt = 0; // Add all variables of datmo
  ROS_INFO("RESET ALL!!!");
  }
}



int remember_id;

Datmo::Datmo(){
  ros::NodeHandle n; 
  ros::NodeHandle n_private("~");
  ROS_INFO("Starting Detection And Tracking of Moving Objects");

  n_private.param("lidar_frame", lidar_frame, string("base_link"));
  n_private.param("world_frame", world_frame, string("map"));
  ROS_INFO("The lidar_frame is: %s and the world frame is: %s", lidar_frame.c_str(), world_frame.c_str());
  n_private.param("threshold_distance", dth, 0.2);
  n_private.param("max_cluster_size", max_cluster_size, 360);
  n_private.param("euclidean_distance", euclidean_distance, 0.25);
  n_private.param("pub_markers", p_marker_pub, false);
  n_private.param("min_cluster_size", min_cluster_size, 0);
  pub_tracks_box_kf     = n.advertise<datmo::TrackArray>("datmo/box_kf", 10);
  pub_marker_array   = n.advertise<visualization_msgs::MarkerArray>("datmo/marker_array", 10);
  sub_scan = n.subscribe("/scan", 1000, &Datmo::callback, this);
  count_sign = n.subscribe("/call_count_sign", 1, callbacksign);
  is_avd = n.subscribe("/is_avoid_obj", 1, callbackobj);
  btn_status = n.subscribe("/bt4_status", 1, btnCallback);

  n_private.param("min_pointsize_obstacle", min_pointsize_obstacle, 5);
  n_private.param("min_height_car_region", min_height_car_region, 0.3);
  n_private.param("max_height_car_region", max_height_car_region, 2.0);
  n_private.param("min_width_car_region", min_width_car_region, -0.2);
  n_private.param("max_width_car_region", max_width_car_region, 2.0);

  n_private.param("max_pointsize_sign", max_pointsize_sign, 15);
  n_private.param("min_width_sign_region", min_width_sign_region, 0.0);
  n_private.param("max_width_sign_region", max_width_sign_region, 1.0);
  n_private.param("min_height_sign_region", min_height_sign_region, 0.3);
  n_private.param("max_height_sign_region", max_height_sign_region, 0.9);
  
  speed = n.advertise<std_msgs::Float32>("/set_speed", 10);

  n_private.param("euclidean_distance_obstacle_avoidance", euclidean_distance_obstacle_avoidance, 1.0);

}

Datmo::~Datmo(){
}

void Datmo::callback(const sensor_msgs::LaserScan::ConstPtr& scan_in){

  // ROS_INFO("I heard: [%d]", count_sign_clt);

  // delete all Markers 
  visualization_msgs::Marker marker;
  visualization_msgs::MarkerArray markera;
  marker.action =3;
  markera.markers.push_back(marker);
  pub_marker_array.publish(markera);


  // ROS_INFO("NUMBER CLUSTERS: [%d]", clusters.size());

  // Only if there is a transform between the world and lidar frame continue
  if(tf_listener.canTransform(world_frame, lidar_frame, ros::Time())){

    //Find position of ego vehicle in world frame, so it can be fed through to the cluster objects
    tf::StampedTransform ego_pose;
    tf_listener.lookupTransform(world_frame, lidar_frame, ros::Time(0), ego_pose);
    
    //TODO implement varying calculation of dt
    dt = 0.08;

    if (time > ros::Time::now()){clusters.clear();}
    time = ros::Time::now();
    auto start = chrono::steady_clock::now();

    vector<pointList> point_clusters_not_transformed;
    Datmo::Clustering(scan_in, point_clusters_not_transformed);

    //Transform Clusters to world_frame
    vector<pointList> point_clusters;
    for (unsigned int i = 0; i < point_clusters_not_transformed.size(); ++i) {
      pointList point_cluster;
      transformPointList(point_clusters_not_transformed[i], point_cluster);
      point_clusters.push_back(point_cluster);
    }


    // Cluster Association based on the Euclidean distance
    // I should check first all the distances and then associate based on the closest distance

    vector<bool> g_matched(point_clusters.size(),false);   // The Group has been matched with a Cluster
    vector<bool> c_matched(clusters.size(),false); // The Cluster object has been matched with a group

    // ROS_INFO("SIZE: %d %d",point_clusters.size(),clusters.size());
    double euclidean[point_clusters.size()][clusters.size()]; // Matrix object to save the euclidean distances

    //Finding mean coordinates of group and associating with cluster Objects
    double mean_x = 0, mean_y = 0;

    for(unsigned int g = 0; g<point_clusters.size();++g){
      double sum_x = 0, sum_y = 0;
        
      for(unsigned int l =0; l<point_clusters[g].size(); l++){
        sum_x = sum_x + point_clusters[g][l].first;
        sum_y = sum_y + point_clusters[g][l].second;
        // ROS_INFO("POINT CLUSTERS [%d] INFO: [%f %f]",g,point_clusters[g][l].first,point_clusters[g][l].second); // Dat Vu add
      }
      mean_x = sum_x / point_clusters[g].size();
      mean_y = sum_y / point_clusters[g].size();

      for(unsigned int c=0;c<clusters.size();++c){
        euclidean[g][c] = abs( mean_x - clusters[c].meanX()) + abs(mean_y - clusters[c].meanY()); 
      }
    }

    //Find the smallest euclidean distance and associate if smaller than the threshold 
    vector<pair <int,int>> pairs;
    for(unsigned int c=0; c<clusters.size();++c){
      unsigned int position;
      double min_distance = euclidean_distance;
      for(unsigned int g=0; g<point_clusters.size();++g){
    if(euclidean[g][c] < min_distance){
      min_distance = euclidean[g][c];
      position = g;
    }
      }
      if(min_distance < euclidean_distance){
        g_matched[position] = true, c_matched[c] = true;
        pairs.push_back(pair<int,int>(c,position));
      }
    }

    /// Dat Vu add on 20/11

    ////////////////////////////////////////////////////////////////////////////////

    //Update Tracked Clusters
    #pragma omp parallel for
    for(unsigned int p=0; p<pairs.size();++p){
      // ROS_INFO("ID CLUSTER: %d %d",clusters[pairs[p].first].id,clusters[pairs[p].first].age);
      clusters[pairs[p].first].update(point_clusters[pairs[p].second], dt, ego_pose,point_clusters[pairs[p].second].size()); // Update vi tri cua cluster hien tai 
    }
       
    //Delete Not Associated Clusters
    unsigned int o=0;
    unsigned int p = clusters.size();
    while(o<p){
      if(c_matched[o] == false){

        std::swap(clusters[o], clusters.back());
        clusters.pop_back();

        std::swap(c_matched[o], c_matched.back());
        c_matched.pop_back();

        o--;
        p--;
      }
    o++;
    }
    


    // Initialisation of new Cluster Objects
    for(unsigned int i=0; i<point_clusters.size();++i){
      if(g_matched[i] == false && point_clusters[i].size()< max_cluster_size && point_clusters[i].size() > min_cluster_size){
        Cluster cl(cclusters, point_clusters[i], dt, world_frame, ego_pose,point_clusters[i].size());
        // ROS_INFO("condicate : %f %f ",cl.meanX(),cl.meanY());
        cclusters++;
        clusters.push_back(cl);
      } 
    }

    // ROS_INFO("CHECK SIGN = %d", checkSign(clusters, max_pointsize_sign,  min_width_sign_region,  max_width_sign_region,  min_height_sign_region,  max_height_sign_region));
    int status_sign = checkSign(clusters, max_pointsize_sign,  min_width_sign_region,  max_width_sign_region,  min_height_sign_region,  max_height_sign_region);
    if(status_sign == 1){
        // SE TRUYEN TIN NHAN NHAN BIEN BAO 
        // msg_sign.data= 1;
        // sign_detect.publish(1);
        ROS_INFO("DETECT CLUSTER LIKE SIGN");
        std_msgs::Int8 msg_sign_test;
        msg_sign_test.data = 1;
        sign_detect.publish(msg_sign_test);
    }


    // Dat Vu test for SSD
    // Cluster myCluster = TrackingSign_SSD(clusters, max_pointsize_sign,  min_width_sign_region,  max_width_sign_region,  min_height_sign_region,  max_height_sign_region);
    // ROS_INFO("My Cluster ID = %d",myCluster.id,myCluster.meanY());

    // Cluster mycluster = clusters.back(); // Get last cluster which we push 
    // clusters.pop_back();



    // for(unsigned int i = 0; i < clusters.size();++i){
    //     ROS_INFO("CLUSTER %d contains: [%f %f]",i,clusters[i].meanX(),clusters[i].meanY());
    // }


    //Visualizations and msg publications
    visualization_msgs::MarkerArray marker_array;
    // datmo::TrackArray track_array_box_kf; 

    // for (unsigned int i =0; i<clusters.size();i++){

    //   // track_array_box_kf.tracks.push_back(clusters[i].msg_track_box_kf);

    //   //TRADINH ADD TO FIND NEAREST CLUSTER  (190-195)
    
      
    //   if (p_marker_pub){
    //     marker_array.markers.push_back(clusters[i].getClosestCornerPointVisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getBoundingBoxCenterVisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getArrowVisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getThetaL1VisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getThetaL2VisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getThetaBoxVisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getClusterVisualisationMessage());
    //       //  marker_array.markers.push_back(clusters[i].getBoundingBoxVisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getBoxModelKFVisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getLShapeVisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getLineVisualisationMessage());
    //     // marker_array.markers.push_back(clusters[i].getBoxSolidVisualisationMessage());
    //   }; 
    // }
    //DRAW MIN CLUSTER
    

    if(clusters.size()> 0){

      // TRUONG HOP TRACKING ID KHONG DAT HIEU QUA


      // int nearest_cluster=findClusterNearestWithCondition(clusters);
      // if(is_obj_detect == false){
      //   if(checkObjectBeforeCar(clusters)){
      //     std_msgs::Int8 msg;
      //     msg.data = 1;
      //     objects_detect.publish(msg);
      //     // ROS_INFO("hello = %d", msg.data);
      //     ROS_INFO("da nhanh dien????????????");
      //     is_obj_detect = true;
      //   }

      //   //CAN THEM DIEU KIEN COUNT SIGN KHI CHAY THAT
      // }
  //     if(is_obj_detect == true){
	// if(nearest_cluster !=-1){
  //       if(calEuclidDistance(clusters[nearest_cluster].meanX(),clusters[nearest_cluster].meanY()) > 2.5){
  //         marker_array.markers.push_back(clusters[nearest_cluster].getBoundingBoxCenterVisualisationMessage());
  //         ROS_INFO("DUNG TRANH VAT CAN!!!!!!!!!!!!!!!!!!");
  //         std_msgs::Int8 msg_avoid;
  //         msg_avoid.data= 1;
  //         objects_detect_avoid.publish(msg_avoid);
  //         // ROS_INFO("hello = %d", msg_avoid.data);
  //         is_obj_detect = false;
  //       }
  //     }else{
	// ROS_INFO("DUNG TRANH VAT CAN !!!!!!!!!!!!!!!!!!!!");
	//  std_msgs::Int8 msg_avoid;
  //         msg_avoid.data= 1;
  //         objects_detect_avoid.publish(msg_avoid);
  //         // ROS_INFO("hello = %d", msg_avoid.data);
  //         is_obj_detect = false;
        
	// }
  //     }

  //                      KET THUC O DAY NH
          
    // for(int i =0;i< clusters.size();i++){

    //   if(clusters[i].pointsize > 5 && clusters[i].meanX()>-2 && clusters[i].meanX()<-0.3 && clusters[i].meanY()>-0.2 && clusters[i].meanY()<2){
    //     marker_array.markers.push_back(clusters[i].getBoundingBoxCenterVisualisationMessage());
    //   }
    // }
      // Get id 
      // int id = clusters[tradinh_i].id;
      // for(int i =0 ; i < point_clusters.size();i++){
      //   ROS_INFO("LENGTH: %d", point_clusters[i].size());
      // }
      // ROS_INFO("LENGTH: %d",clusters[tradinh_i].pointsize);
      // ROS_INFO("ID: %d", clusters[tradinh_i].id);
      ROS_INFO("COUNT SIGN: %d",count_sign_clt);

      // int check_obj = checkObjectBeforeCar(clusters);
      
      // ROS_INFO("TRA DINH NGOI CODE VAO SANG SOM NGAY DAU THANG 12 : %d", check_obj);
      // if(check_obj != -1){
      //     remember_id = check_obj;
      // for(int i=0;i<clusters.size();i++){
      // if(clusters[i].id == remember_id){
      //       marker_array.markers.push_back(clusters[i].getBoundingBoxCenterVisualisationMessage());
      //       ROS_INFO("ID remember: %d", clusters[i].id);
      // }
      // }

      //OPEN COMMENT HERE 
      // int check_obj = -1;
      // for(int i =0;i< clusters.size();i++){
      //   double x = clusters[i].closest_corner_point.first;
      //   double y = clusters[i].closest_corner_point.second;
      //   if( ((clusters[i].length_box >0.25 && clusters[i].length_box <0.37)|| (clusters[i].width_box > 0.25 && clusters[i].width_box<0.37)) && ((clusters[i].closest_corner_point.first > -3 && clusters[i].closest_corner_point.first < -0.3 && 
      //       clusters[i].closest_corner_point.second >= -0.2 && clusters[i].closest_corner_point.second <= 0.2) || 
      //       (calEuclidDistance(x,y-0.2) < 3 && atan(-x/(y-0.2)) > PI/4 && atan(-x/(y-0.2)) < PI/2 && x < 0))
      //       && clusters[i].pointsize > min_pointsize_obstacle){
      //           marker_array.markers.push_back(clusters[i].getBoundingBoxCenterVisualisationMessage());
      //           marker_array.markers.push_back(clusters[i].getClosestCornerPointVisualisationMessage());
      //           marker_array.markers.push_back(clusters[i].getClusterVisualisationMessage());
      //           // check_obj = clusters[i].id;
      //           ROS_INFO("arctan: %f",atan(-x/(y-0.2)));
      //           ROS_INFO("[x,y]: [%f,%f]",x,y);
      //           ROS_INFO("euclid: %f",calEuclidDistance(x,y-0.2));
      //           ROS_INFO("lengh, width  : [%f,%f]",clusters[i].length_box,clusters[i].width_box);
      //           std_msgs::Int8 msg;
      //           msg.data = 1;
      //           objects_detect.publish(msg);
      //           ROS_INFO("TRUYEN TIN NHAN TRANH VAT CAN : %d", msg.data);
      //           // marker_array.markers.push_back(clusters[i].getClosestCornerPointVisualisationMessage());
      //           // ROS_INFO("da nhanh dien");
      //           wait(3);
      //           std_msgs::Int8 msg_avoid;
      //           msg_avoid.data= 1;
      //           objects_detect_avoid.publish(msg_avoid);
      //           ROS_INFO("TRUEY TIN NHAN DUNG TRANH VAT CAN SAU 3 S: %d", msg_avoid.data);
      //           remember_id = check_obj;
      //           break;
      //       }
      //     }




      // ROS_INFO("IS OBJECT DETECT : %d ", is_obj_detect );
      if(is_obj_detect ==false && count_sign_clt == 2){
        // ROS_INFO("da nhanh dien");
        int check_obj = findObjectBeforeCar3(clusters,min_pointsize_obstacle,min_height_car_region, max_height_car_region, min_width_car_region,  max_width_car_region);
        // ROS_INFO("TRA DINH NGOI CODE VAO SANG SOM NGAY DAU THANG 12 : %d", check_obj);
        if(check_obj != -1){
          // TRUYEN TIN NHAN NHAN DIEN VA TRANH VAT CAN 
          std_msgs::Int8 msg;
          msg.data = 1;
          objects_detect.publish(msg);
          ROS_INFO("TRUYEN TIN NHAN TRANH VAT CAN : %d", msg.data);
          // marker_array.markers.push_back(clusters[i].getClosestCornerPointVisualisationMessage());
          // ROS_INFO("da nhanh dien");
          wait(3);
          std_msgs::Int8 msg_avoid;
          msg_avoid.data= 1;
          objects_detect_avoid.publish(msg_avoid);
          ROS_INFO("TRUEY TIN NHAN DUNG TRANH VAT CAN SAU 3 S: %d", msg_avoid.data);
          remember_id = check_obj;
          is_obj_detect = true;
        }
      }

      // if(is_obj_detect == true){
      //   bool check_for_loop = false;
      //   for(int i=0;i<clusters.size();i++){
      //     if(clusters[i].id == remember_id){
      //       // marker_array.markers.push_back(clusters[i].getBoundingBoxCenterVisualisationMessage());
      //       ROS_INFO("ID remember: %d", clusters[i].id);
	    //       ROS_INFO("DISTANCE  : %f", calEuclidDistance(clusters[i].meanX(),clusters[i].meanY()));
      //       check_for_loop = true;
      //       ROS_INFO("MATDO DIEM : %d", clusters[i].pointsize);
      //       // ROS_INFO("dien tich : %f",clusters[i].length_box*clusters[i].width_box);
      //       ROS_INFO("LENGTH : %f", clusters[i].length_box);
      //       ROS_INFO("WIDTH : %f", clusters[i].width_box);
      //       ROS_INFO("4 POINT CONNER : [%f,%f]",clusters[i].closest_corner_point.first,clusters[i].closest_corner_point.second);
      //       ROS_INFO("mean : [%f,%f]",clusters[i].meanX(),clusters[i].meanY());
      //       if(calEuclidDistance(clusters[i].meanX(),clusters[i].meanY())> euclidean_distance_obstacle_avoidance && clusters[i].meanX() > 0 ){
      //         // SE TRUYEB TIN NHAN QUAY TRO LAI LAN DUONG
              
      //         std_msgs::Int8 msg_avoid;
      //         msg_avoid.data= 1;
      //         objects_detect_avoid.publish(msg_avoid);
      //         //marker_array.markers.push_back(clusters[i].getBoundingBoxCenterVisualisationMessage());
      //         //marker_array.markers.push_back(clusters[i].getClosestCornerPointVisualisationMessage());
      //         ROS_INFO("dien tich : %f",clusters[i].length_box*clusters[i].width_box);
      //         ROS_INFO("TRUEY TIN NHAN DUNG TRANH VAT CAN : %d", msg_avoid.data);
      //         is_obj_detect = false;
      //         // remember_id = -1;
      //         break;
      //       }
      //     }
      //   }
      //   if(check_for_loop == false && checkRightCar(clusters) == false){
      //     std_msgs::Int8 msg_avoid;
      //     msg_avoid.data= 1;
      //     objects_detect_avoid.publish(msg_avoid);
      //     ROS_INFO("TRUEY TIN NHAN DUNG TRANH VAT CAN IN CHECK LOOP FALSE: %d", msg_avoid.data);
      //     is_obj_detect = false;
      //   }    
      // }








      //TRA DINH CODE VAO MOT BUOI TRUA DAU THANG 12



      // ROS_INFO("NEAREST CENTER = [%f %f]",clusters[tradinh_i].meanX(),clusters[tradinh_i].meanY());
   

      // objects_detect = n.advertise<std_msgs::Int8>("/objects_detection_by_cluster", 1);
      // ROS_INFO("SIZE: %d %d",point_clusters.size(),clusters.size();
      // marker_array.markers.push_back(getCenter(p_tra));

      pub_marker_array.publish(marker_array);
    }
    // pub_tracks_box_kf.publish(track_array_box_kf);
    // visualiseGroupedPoints(point_clusters);
    
  }
  else{ //If the tf is not possible init all states at 0
    ROS_WARN_DELAYED_THROTTLE(1 ,"No transform could be found between %s and %s", lidar_frame.c_str(), world_frame.c_str());
  };
}



void Datmo::visualiseGroupedPoints(const vector<pointList>& point_clusters){
  //Publishing the clusters with different colors
  visualization_msgs::MarkerArray marker_array;
  //Populate grouped points message
  visualization_msgs::Marker gpoints;
  gpoints.header.frame_id = world_frame;
  gpoints.header.stamp = ros::Time::now();
  gpoints.ns = "clustered_points";
  gpoints.action = visualization_msgs::Marker::ADD;
  gpoints.pose.orientation.w = 1.0;
  gpoints.type = visualization_msgs::Marker::POINTS;
  // POINTS markers use x and y scale for width/height respectively
  gpoints.scale.x = 0.04;
  gpoints.scale.y = 0.04;
  for(unsigned int i=0; i<point_clusters.size(); ++i){

    gpoints.id = cg;
    cg++;
    gpoints.color.g = rand() / double(RAND_MAX);
    gpoints.color.b = rand() / double(RAND_MAX);
    gpoints.color.r = rand() / double(RAND_MAX);
    gpoints.color.a = 1.0;
    //gpoints.lifetime = ros::Duration(0.08);
    for(unsigned int j=0; j<point_clusters[i].size(); ++j){
      geometry_msgs::Point p;
      p.x = point_clusters[i][j].first;
      p.y = point_clusters[i][j].second;
      p.z = 0;
      gpoints.points.push_back(p);
    }
    marker_array.markers.push_back(gpoints);
    gpoints.points.clear();
  }
  pub_marker_array.publish(marker_array);

}


void Datmo::Clustering(const sensor_msgs::LaserScan::ConstPtr& scan_in, vector<pointList> &clusters){
  scan = *scan_in;


  int cpoints = 0;
  
  //Find the number of non inf laser scan values and save them in c_points
  for (unsigned int i = 0; i < scan.ranges.size(); ++i){
    if(isinf(scan.ranges[i]) == 0){
      cpoints++;
    }
  }
  const int c_points = cpoints;

  int j = 0;
  vector< vector<float> > polar(c_points +1 ,vector<float>(2)); //c_points+1 for wrapping
  for(unsigned int i = 0; i<scan.ranges.size(); ++i){
    if(!isinf(scan.ranges[i])){
      polar[j][0] = scan.ranges[i]; //first column is the range 
      polar[j][1] = scan.angle_min + i*scan.angle_increment; //second angle in rad
      j++;
    }
  }

  //Complete the circle
  polar[c_points] = polar[0];

  //Find clusters based on adaptive threshold distance
  float d;

 //There are two flags, since two consecutive points can belong to two independent clusters
  vector<bool> clustered1(c_points+1 ,false); //change to true when it is the first of the cluster
  vector<bool> clustered2(c_points+1 ,false); // change to true when it is clustered by another one

  float l = 45; // λ is an acceptable angle for determining the points to be of the same cluster
  l = l * 0.0174532;   // degree to radian conversion;
  const float s = 0;   // σr is the standard deviation of the noise of the distance measure
  for (unsigned int i=0; i < c_points ; ++i){
    double dtheta = polar[i+1][1]- polar[i][1];
    double adaptive = min(polar[i][0],polar[i+1][0]) * (sin(dth)) / (sin(l - (dth))) + s; //Dthreshold
    d = sqrt( pow(polar[i][0],2) + pow(polar[i+1][0],2)-2 * polar[i][0]*polar[i+1][0]*cos(polar[i+1][1] - polar[i][1]));
    //ROS_INFO_STREAM("distance: "<<dth<<", adapt: "<<adaptive<<", dtheta: "<<dtheta);
    //if(polar[i+1][1]- polar[i][1]<0){
      //ROS_INFO_STREAM("problem");
    //}

    if(d<dth) {
      clustered1[i] = true; //both points belong to clusters
      clustered2[i+1] = true;}
  }

  clustered2[0] = clustered2[c_points];
  
  //Going through the points and finding the beginning of clusters and number of points
  vector<int> begin; //saving the first index of a cluster
  vector<int> nclus; //number of clustered points
  int i =0;
  bool flag = true; // flag for not going back through the stack 

  while(i<c_points && flag==true){

    if (clustered1[i] == true && clustered2[i] == false && flag == true){
      begin.push_back(i);
      nclus.push_back(1);
      while(clustered2[i+1] == true && clustered1[i+1] == true ){
	i++;
	++nclus.back();
	if(i==c_points-1 && flag == true){
	  i = -1;
	  flag = false;
	}
      }
      ++nclus.back();//take care of 0 1 flags - last of the cluster
    }
  i++;
  }
  // take care of last point being beginning of cluster
  if(clustered1[cpoints-1]== true and clustered2[c_points-1] == false){
      begin.push_back(cpoints-1);
      nclus.push_back(1);
      i = 0;
      while(clustered2[i] == true && clustered1[i] == true ){
	i++;
	++nclus.back();
      }

  }

  polar.pop_back(); //remove the wrapping element
  int len = polar.size();

  for(unsigned int i=0; i<begin.size(); ++i){

    pointList cluster;

    double x,y;
    int j =begin[i];
    bool fl = true; // flag for not going back through the stack 

    while (j<nclus[i]+begin[i]){
      if(j== len && fl == true) fl = false;
      if (fl == true)
      {
        x = polar[j][0] * cos(polar[j][1]);       //x = r × cos( θ )
        y = polar[j][0] * sin(polar[j][1]);       //y = r × sin( θ )
      }
      else{
       x = polar[j-len][0] *cos(polar[j-len][1]); //x = r × cos( θ )
       y = polar[j-len][0] *sin(polar[j-len][1]); //y = r × sin( θ ) 
      }
      cluster.push_back(Point(x, y));
      ++j;
    }
    clusters.push_back(cluster);
  }
}
void Datmo::transformPointList(const pointList& in, pointList& out){
  //This funcion transforms pointlist between coordinate frames and it is a wrapper for the
  //transformPoint function
  //There is not try catch block because it is supposed to be already encompassed into one
  
  geometry_msgs::PointStamped point_in, point_out;
  Point point; 
  point_in.header.frame_id = lidar_frame;
  point_in.header.stamp = ros::Time(0);
  for (unsigned int i = 0; i < in.size(); ++i) {
    point_in.point.x = in[i].first;
    point_in.point.y = in[i].second;
    tf_listener.transformPoint(world_frame, point_in , point_out);
    point.first = point_out.point.x;
    point.second= point_out.point.y;
    out.push_back(point);
  }
}
