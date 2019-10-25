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
struct Control {
	int x;
	int y;
	bool mouse_click;
	float rotx;
	float roty;
	double gx;
	double gy;
	double gz;

	Control() :x(0), y(0), rotx(0), roty(0), mouse_click(false),
		gx(0), gy(0), gz(0) {

	}
};

struct Object_input
{
	bool is_found;
	pcl::PointCloud<pcl::PointNormal> pointcloud;
	pcl::PointCloud<pcl::Normal> normal;
	cv::Mat silhouette;
	cv::Mat depth;
	Vector3 center;

	void Init(const int W, const int H)
	{
		is_found = false;
		silhouette = cv::Mat(cv::Size(W, H), CV_8UC1, cv::Scalar(0));
		depth = cv::Mat(cv::Size(W, H), CV_16UC1, cv::Scalar(0));
		center = Vector3::Zero();
		pointcloud.points.clear();
		pointcloud.points.reserve(W*H);
	}

	void operator=(const Object_input& obj_input)
	{
		is_found = obj_input.is_found;
		pointcloud.points.assign(obj_input.pointcloud.points.begin(), obj_input.pointcloud.points.end());
		silhouette = obj_input.silhouette.clone();
		depth = obj_input.depth.clone();
		center = obj_input.center;
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

	void operator=(const Image_InputData& image_input)
	{
		width = image_input.width;
		height = image_input.height;

		color = image_input.color.clone();
		depth = image_input.depth.clone();
		silhouette = image_input.silhouette.clone();

		hand = image_input.hand;
		item = image_input.item;
	}
};