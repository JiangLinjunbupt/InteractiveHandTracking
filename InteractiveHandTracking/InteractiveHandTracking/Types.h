#pragma once
#include<pcl\point_types.h>
#include<pcl\filters\voxel_grid.h>
#include<pcl\filters\statistical_outlier_removal.h>
//���������ǰ�棬��֪��Ϊʲô����opencv����ͻ��޷�ʹ��statistical_outlier_removal����������۶����¼���
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
	int pointcloud_idx;

	Vector3 correspond;
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

///��ʼ���彻�������������Ϣ�ṹ�� �� ����������Ϣ�ṹ��
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

	int *idxs_image;

	Object_input hand;
	Object_input item;

	void Init(const int W, const int H)
	{
		width = W;
		height = H;

		color = cv::Mat(cv::Size(width, height), CV_8UC3, cv::Scalar(0, 0, 0));
		depth = cv::Mat(cv::Size(width, height), CV_16UC1, cv::Scalar(0));
		silhouette = cv::Mat(cv::Size(width, height), CV_8UC1, cv::Scalar(0));

		idxs_image = new int[W*H]();
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

		std::copy(image_input.idxs_image, image_input.idxs_image + width * height, idxs_image);
	}
};

struct Glove_InputData
{
	VectorN params;
	VectorN shapeparams;

	void Init()
	{
		params = VectorN::Zero(NUM_HAND_POSE_PARAMS);
		shapeparams = VectorN::Zero(NUM_HAND_SHAPE_PARAMS);
	}
};

struct InputData
{
	Image_InputData image_data;
	Glove_InputData glove_data;

	void Init(const int W, const int H)
	{
		image_data.Init(W, H);
		glove_data.Init();
	}
};


struct Rendered_Images
{
	cv::Mat total_silhouette;
	cv::Mat total_depth;

	cv::Mat rendered_object_silhouette;
	cv::Mat rendered_object_depth;

	cv::Mat rendered_hand_silhouette;
	cv::Mat rendered_hand_depth;

	void init(int w, int h)
	{
		total_silhouette = cv::Mat(cv::Size(w, h), CV_8UC1, cv::Scalar(0));
		rendered_object_silhouette = cv::Mat(cv::Size(w, h), CV_8UC1, cv::Scalar(0));
		rendered_hand_silhouette = cv::Mat(cv::Size(w, h), CV_8UC1, cv::Scalar(0));

		total_depth = cv::Mat(cv::Size(w, h), CV_16UC1, cv::Scalar(0));
		rendered_object_depth = cv::Mat(cv::Size(w, h), CV_16UC1, cv::Scalar(0));
		rendered_hand_depth = cv::Mat(cv::Size(w, h), CV_16UC1, cv::Scalar(0));
	}

	void setToZero()
	{
		total_silhouette.setTo(0);
		rendered_object_silhouette.setTo(0);
		rendered_hand_silhouette.setTo(0);

		total_depth.setTo(0);
		rendered_object_depth.setTo(0);
		rendered_hand_depth.setTo(0);
	}

	void operator=(Rendered_Images& rendered_images)
	{
		rendered_images.total_silhouette.copyTo(total_silhouette);
		rendered_images.total_depth.copyTo(total_depth);

		rendered_images.rendered_hand_silhouette.copyTo(rendered_hand_silhouette);
		rendered_images.rendered_hand_depth.copyTo(rendered_hand_depth);

		rendered_images.rendered_object_depth.copyTo(rendered_object_depth);
		rendered_images.rendered_object_silhouette.copyTo(rendered_object_silhouette);
	}
};