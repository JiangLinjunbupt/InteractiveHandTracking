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

#include"LogUtils.h"

using namespace std;
using namespace std::chrono;

#define M_PI 3.14159265358979323846

#define ANGLE_TO_RADIUS 0.017453
#define RADIUS_TO_ANGLE 57.2974

#define NUM_OBJECT_PARAMS 6
#define NUM_HAND_FINGER_PARAMS 45
#define NUM_HAND_POSE_PARAMS 51
#define NUM_HAND_SHAPE_PARAMS 10

#define NUM_HAND_GLOBAL_PARAMS 6
#define NUM_HAND_WRIST_PARAMS 3
#define NUM_HAND_POSITION_PARAMS 3



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

/// Linear system lhs*x=rhs
struct LinearSystem {
	Matrix_MxN lhs; // J^T*J
	VectorN rhs; // J^T*r
	LinearSystem() {}
	LinearSystem(int n) {
		lhs = Matrix_MxN::Zero(n, n);
		rhs = VectorN::Zero(n);
	}
};

struct DataAndCorrespond
{
	Vector3 pointcloud;
	Vector3 pointcloud_n;
	int pointcloud_idx;

	Vector3 correspond;
	Vector3 correspond_n;
	int correspond_idx;

	bool is_match;
};

enum Object_type { yellowSphere, redCube };

enum RuntimeType
{
	REALTIME,Dataset_MSRA_14, Dataset_MSRA_15, Handy_teaser, ICVL, NYU, GeneratedData, Guess_who
};

enum FingerType
{
	Index, Middle, Pinky, Ring, Thumb
};
struct Collision
{
	int id;
	bool root;

	Eigen::Vector3f Init_Center;
	Eigen::Vector3f Update_Center;

	float Radius;
	int joint_belong;
	FingerType fingerType;

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
	const int LOSS_DETECT_THRESHOLD = 20;

	pcl::PointCloud<pcl::PointNormal> pointcloud;
	cv::Mat silhouette;
	cv::Mat depth;
	Vector3 center;

	int not_detect_count;
	bool now_detect;
	bool pre_detect;
	bool first_detect;
	bool loss_detect;

	void Init(const int W, const int H)
	{
		silhouette = cv::Mat(cv::Size(W, H), CV_8UC1, cv::Scalar(0));
		depth = cv::Mat(cv::Size(W, H), CV_16UC1, cv::Scalar(0));
		center = Vector3::Zero();
		pointcloud.points.clear();
		pointcloud.points.reserve(W*H);

		not_detect_count = 0;
		now_detect = false;
		pre_detect = false;
		first_detect = false;
		loss_detect = true;
	}

	void operator=(const Object_input& obj_input)
	{
		pointcloud.points.assign(obj_input.pointcloud.points.begin(), obj_input.pointcloud.points.end());
		silhouette = obj_input.silhouette.clone();
		depth = obj_input.depth.clone();
		center = obj_input.center;

		not_detect_count = obj_input.not_detect_count;
		now_detect = obj_input.now_detect;
		pre_detect = obj_input.pre_detect;
		first_detect = obj_input.first_detect;
		loss_detect = obj_input.loss_detect;
	}

	void UpdateStatus(bool is_detect)
	{
		pre_detect = now_detect;

		if (is_detect)
		{
			not_detect_count = 0;
			now_detect = true;
			loss_detect = false;
		}
		else
		{
			not_detect_count++;
			now_detect = false;
			if (not_detect_count > LOSS_DETECT_THRESHOLD)
				loss_detect = true;
		}
	}

	void ClearPointcloudAndCenter()
	{
		pointcloud.points.clear();
		center.setZero();
	}
};

struct Image_InputData
{
	int width;
	int height;

	cv::Mat color;
	cv::Mat depth;
	cv::Mat silhouette;

	int *idxs_image;

	Object_input hand;
	vector<Object_input> item;

	void Init(const int W, const int H,int object_num)
	{
		width = W;
		height = H;

		color = cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar(0, 0, 0));
		depth = cv::Mat(cv::Size(width, height), CV_16UC1, cv::Scalar(0));
		silhouette = cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0));

		idxs_image = new int[W*H]();
		hand.Init(W, H);

		for (int obj_id = 0; obj_id < object_num; ++obj_id)
		{
			Object_input tmpObject;
			tmpObject.Init(W, H);
			item.emplace_back(tmpObject);
		}
	}

	void operator=(const Image_InputData& image_input)
	{
		width = image_input.width;
		height = image_input.height;

		color = image_input.color.clone();
		depth = image_input.depth.clone();
		silhouette = image_input.silhouette.clone();

		hand = image_input.hand;

		for (size_t obj_id = 0; obj_id < image_input.item.size(); ++obj_id)
			item[obj_id] = image_input.item[obj_id];

		std::copy(image_input.idxs_image, image_input.idxs_image + width * height, idxs_image);
	}
};

struct Glove_InputData
{
	VectorN params;
	VectorN pre_params;
	void Init()
	{
		params = VectorN::Zero(NUM_HAND_POSE_PARAMS);
		pre_params = VectorN::Zero(NUM_HAND_POSE_PARAMS);
	}
};

struct InputData
{
	Image_InputData image_data;
	Glove_InputData glove_data;

	void Init(const int W, const int H,int object_num)
	{
		image_data.Init(W, H, object_num);
		glove_data.Init();
	}
};


struct Rendered_Images
{
	cv::Mat rendered_object_silhouette;
	cv::Mat rendered_object_depth;

	cv::Mat rendered_hand_silhouette;
	cv::Mat rendered_hand_depth;

	void init(int w, int h)
	{
		rendered_object_silhouette = cv::Mat(cv::Size(w, h), CV_8UC1, cv::Scalar(0));
		rendered_hand_silhouette = cv::Mat(cv::Size(w, h), CV_8UC1, cv::Scalar(0));

		rendered_object_depth = cv::Mat(cv::Size(w, h), CV_16UC1, cv::Scalar(0));
		rendered_hand_depth = cv::Mat(cv::Size(w, h), CV_16UC1, cv::Scalar(0));
	}

	void setToZero()
	{
		rendered_object_silhouette.setTo(0);
		rendered_hand_silhouette.setTo(0);

		rendered_object_depth.setTo(0);
		rendered_hand_depth.setTo(0);
	}

	void operator=(Rendered_Images& rendered_images)
	{
		rendered_hand_silhouette = rendered_images.rendered_hand_silhouette.clone();
		rendered_hand_depth = rendered_images.rendered_hand_depth.clone();

		rendered_object_depth = rendered_images.rendered_object_depth.clone();
		rendered_object_silhouette = rendered_images.rendered_object_silhouette.clone();
	}
};