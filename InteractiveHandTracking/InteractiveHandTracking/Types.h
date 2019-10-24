#pragma once
#include<pcl\point_types.h>
#include<pcl\filters\voxel_grid.h>
#include<pcl\filters\statistical_outlier_removal.h>
//必须放在最前面，不知道为什么放在opencv后面就会无法使用statistical_outlier_removal这个方法（幺蛾子事件）
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <limits>
#include <set>
#include <map>
#include<string>
#include<chrono>

using namespace std;
using namespace std::chrono;

#define M_PI 3.14159265358979323846

#define ANGLE_TO_RADIUS 0.017453;
#define RADIUS_TO_ANGLE 57.2974;

typedef Eigen::Matrix4f Matrix4;
typedef Eigen::Matrix<float, 3, 3> Matrix3;
typedef Eigen::Matrix<float, 2, 3> Matrix_2x3;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic> Matrix_3xN;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>  Matrix_MxN;
typedef Eigen::VectorXf VectorN;
typedef Eigen::Vector2f Vector2;
typedef Eigen::Vector3f Vector3;


/// Nan for the default type
inline float nan() { return (std::numeric_limits<float>::quiet_NaN)(); }
inline float inf() { return (std::numeric_limits<float>::max)(); }

enum RuntimeType
{
	REALTIME,Dataset_MSRA_14, Dataset_MSRA_15, Handy_teaser, ICVL, NYU, GeneratedData, Guess_who
};

///开始定义交互物体的输入信息结构体 和 人手输入信息结构体

struct Object_input
{
	bool is_found;
	pcl::PointCloud<pcl::PointXYZ> pointcloud;
	cv::Mat silhouette;
	cv::Mat depth;

	void Init(const int W, const int H)
	{
		is_found = false;
		silhouette = cv::Mat(cv::Size(W, H), CV_8UC1, cv::Scalar(0));
		depth = cv::Mat(cv::Size(W, H), CV_16UC1, cv::Scalar(0));
		pointcloud.points.clear();
	}
};

struct Image_InputData
{
	int width;
	int height;

	cv::Mat color;
	cv::Mat depth;
	cv::Mat silhouette;

	Object_input hand;
	Object_input item;

	void Init(const int W, const int H)
	{
		width = W;
		height = H;

		color = cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar(0, 0, 0));
		depth = cv::Mat(cv::Size(width, height), CV_16UC1, cv::Scalar(0));
		silhouette = cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0));

		hand.Init(W, H);
		item.Init(W, H);
	}
};