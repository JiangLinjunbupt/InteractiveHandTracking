#pragma once
#include"Types.h"
#include"Camera.h"

struct Object_attribute
{
	Object_type type;

	//可能的属性
	Vector3 color;

	float radius; //为球体准备的半径
	
	float length;
	Eigen::MatrixXf CornerPoints; //为立方体准备的角点

};
class Interacted_Object
{
protected:
	Camera* mCamera;
	vector<Vector3> init_Vertices;
	vector<Vector3> init_Normal;

	virtual void GenerateOrLoadPointsAndNormal() = 0;
	virtual void UpdateVerticesAndNormal() = 0;

public:
	Obj_status mObj_status;
	Eigen::Matrix4f relative_Trans;
	//运动参数
	Eigen::VectorXf object_params;
	int Num_object_prams;

	//网格参数
	vector<Vector3> Final_Vertices;
	vector<Vector3> Final_Normal;
	vector<Vector3> Face_idx;

	//可见点
	vector<std::pair<Vector3, int>> Visible_2D;
	pcl::PointCloud<pcl::PointNormal> Visible_3D;

	//具体的物体属性
	Object_attribute mObject_attribute;

	//生成的深度图和轮廓图
	cv::Mat generatedDepth;
	cv::Mat generatedSilhouette;
public:
	Interacted_Object(Camera* camera) :mCamera(camera) {
		generatedDepth = cv::Mat(cv::Size(mCamera->width(), mCamera->height()), CV_16UC1, cv::Scalar(0));
		generatedSilhouette = cv::Mat(cv::Size(mCamera->width(), mCamera->height()), CV_8UC1, cv::Scalar(0));

		//所有的物体都是由旋转和平移，一共6个参数组成
		Num_object_prams = 6;
		object_params = Eigen::VectorXf::Zero(6);
	};

	virtual ~Interacted_Object() {}
	virtual void GenerateDepthAndSilhouette() = 0;
	void ClearDepthAndSilhouette()
	{
		this->generatedDepth.setTo(0);
		this->generatedSilhouette.setTo(0);
	}
	virtual void ShowDepth() = 0;
	virtual void Update(const Eigen::VectorXf& params) = 0;

	void object_jacobain(Eigen::MatrixXf& jacobain, int idx)
	{
		jacobain = Eigen::MatrixXf::Zero(3, Num_object_prams);

		//前三个参数是位移
		jacobain.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();

		Eigen::Matrix4f x_jacob, y_jacob, z_jacob;
		RotateMatrix_jacobain(object_params(3), object_params(4), object_params(5), x_jacob, y_jacob, z_jacob);

		Eigen::Vector4f tmp = Eigen::Vector4f(init_Vertices[idx](0), init_Vertices[idx](1), init_Vertices[idx](2), 1);

		jacobain.col(3) = (x_jacob * tmp).head(3);
		jacobain.col(4) = (y_jacob * tmp).head(3);
		jacobain.col(5) = (z_jacob * tmp).head(3);
	}

	virtual float SDF(const Eigen::Vector3f& p) = 0;
	virtual bool Is_inside(const Eigen::VectorXf& p) = 0;
	virtual Eigen::VectorXf FindTarget(const Eigen::VectorXf& p) = 0;
	virtual Eigen::VectorXf FindTouchPoint(const Eigen::VectorXf& p) = 0;
	virtual Eigen::MatrixXf GetObjectTransMatrix() = 0;

protected:
	Eigen::Matrix3f EularToRotateMatrix(float x, float y, float z)
	{
		Eigen::Matrix3f x_rotate = Eigen::Matrix3f::Identity();
		Eigen::Matrix3f y_rotate = Eigen::Matrix3f::Identity();
		Eigen::Matrix3f z_rotate = Eigen::Matrix3f::Identity();

		float sx = sin(x); float cx = cos(x);
		float sy = sin(y); float cy = cos(y);
		float sz = sin(z); float cz = cos(z);

		x_rotate(1, 1) = cx; x_rotate(1, 2) = -sx;
		x_rotate(2, 1) = sx; x_rotate(2, 2) = cx;

		y_rotate(0, 0) = cy; y_rotate(0, 2) = sy;
		y_rotate(2, 0) = -sy; y_rotate(2, 2) = cy;

		z_rotate(0, 0) = cz; z_rotate(0, 1) = -sz;
		z_rotate(1, 0) = sz; z_rotate(1, 1) = cz;


		return x_rotate*y_rotate*z_rotate;
	}

	void RotateMatrix_jacobain(float x, float y, float z, Eigen::Matrix4f& x_jacob, Eigen::Matrix4f& y_jacob, Eigen::Matrix4f& z_jacob)
	{
		Eigen::Matrix4f x_rotate = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f y_rotate = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f z_rotate = Eigen::Matrix4f::Identity();

		Eigen::Matrix4f x_rotate_jacob = Eigen::Matrix4f::Zero();
		Eigen::Matrix4f y_rotate_jacob = Eigen::Matrix4f::Zero();
		Eigen::Matrix4f z_rotate_jacob = Eigen::Matrix4f::Zero();

		float sx = sin(x); float cx = cos(x);
		float sy = sin(y); float cy = cos(y);
		float sz = sin(z); float cz = cos(z);

		x_rotate(1, 1) = cx; x_rotate(1, 2) = -sx;
		x_rotate(2, 1) = sx; x_rotate(2, 2) = cx;

		y_rotate(0, 0) = cy; y_rotate(0, 2) = sy;
		y_rotate(2, 0) = -sy; y_rotate(2, 2) = cy;

		z_rotate(0, 0) = cz; z_rotate(0, 1) = -sz;
		z_rotate(1, 0) = sz; z_rotate(1, 1) = cz;


		x_rotate_jacob(1, 1) = -sx; x_rotate_jacob(1, 2) = -cx;
		x_rotate_jacob(2, 1) = cx; x_rotate_jacob(2, 2) = -sx;

		y_rotate_jacob(0, 0) = -sy; y_rotate_jacob(0, 2) = cy;
		y_rotate_jacob(2, 0) = -cy; y_rotate_jacob(2, 2) = -sy;

		z_rotate_jacob(0, 0) = -sz; z_rotate_jacob(0, 1) = -cz;
		z_rotate_jacob(1, 0) = cz; z_rotate_jacob(1, 1) = -sz;

		x_jacob = x_rotate_jacob*y_rotate*z_rotate;
		y_jacob = x_rotate*y_rotate_jacob*z_rotate;
		z_jacob = x_rotate*y_rotate*z_rotate_jacob;
	}
};


class YellowSphere : public Interacted_Object
{
protected:
	void GenerateOrLoadPointsAndNormal();
	void UpdateVerticesAndNormal();
public:
	YellowSphere(Camera* camera);
	virtual ~YellowSphere() {};

	void GenerateDepthAndSilhouette();
	void ShowDepth();
	void Update(const Eigen::VectorXf& params);
	float SDF(const Eigen::Vector3f& p);
	bool Is_inside(const Eigen::VectorXf& p);
	Eigen::VectorXf FindTarget(const Eigen::VectorXf& p);
	Eigen::VectorXf FindTouchPoint(const Eigen::VectorXf& p);
	Eigen::MatrixXf GetObjectTransMatrix();
};


class RedCube :public Interacted_Object
{
private:
	Eigen::Matrix<float, 8, 3> FinalCornerPoints;
	Eigen::Matrix<float, 3, 3> Coordinate;
	Eigen::Matrix4f T_local;
	Eigen::Matrix4f T_local_inverse;

protected:
	void GenerateOrLoadPointsAndNormal();
	void UpdateVerticesAndNormal();
public:
	RedCube(Camera* camera);
	virtual ~RedCube() {};

	void GenerateDepthAndSilhouette();
	void ShowDepth();
	void Update(const Eigen::VectorXf& params);
	float SDF(const Eigen::Vector3f& p);
	bool Is_inside(const Eigen::VectorXf& p);
	Eigen::VectorXf FindTarget(const Eigen::VectorXf& p);
	Eigen::VectorXf FindTouchPoint(const Eigen::VectorXf& p);
	Eigen::MatrixXf GetObjectTransMatrix();
};