#pragma once
#include <assert.h>
#include <Eigen/Dense>   //包含所有普通的矩阵函数
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/Core>

#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include"Camera.h"
#include<math.h>
#include<algorithm>
using namespace std;


class HandModel
{
public:

	///////////////////////////////////////
	//*********碰撞相关****************//
	/////////////////////////////////////
	int NumOfCollision;
	std::vector<std::map<float, int>> Joint_related_Vertex;
	std::map<int, float> Joint_Radius;
	std::map<int, int> Finger_Tip; //这里对应着每个指尖的顶点
	std::vector<Collision> Collision_sphere;
	Eigen::MatrixXi Collision_Judge_Matrix;
	////////碰撞相关结束///////////

	//关节点的半径更新，这里使用的是附近权重最大的20个点，距离关节点的最短距离作为半径
	void Compute_JointRadius();
	void Compute_CollisionSphere();
	//CollisionSphere的更新内容放在了LBS_Updata中了

	//然后就是每次计算是否有碰撞，并且根据碰撞点计算雅各比矩阵
	//判断是否碰撞
	void Judge_Collision();

	//每一个碰撞点的雅各比矩阵
	void CollisionPoint_Jacobian(Eigen::MatrixXf& jacobain, int jointBelong, Eigen::Vector3f& point);



	int Joints_num;
	int Vertex_num;
	int Face_num;

	Eigen::MatrixXi F;
	Eigen::MatrixXf F_normal;

	//计算出来的值
	Eigen::MatrixXf V_Final;   //经过pose和trans之后的某个姿态下的顶点
	Eigen::MatrixXf V_Normal_Final;  //顶点的法向量
	Eigen::MatrixXf J_Final;   //经过pose和trans之后某个姿态下的关节点

	std::vector<std::pair<Eigen::Vector2i, int>> V_Visible_2D;   //2D的可见点
	pcl::PointCloud<pcl::PointNormal> Visible_3D;

	std::map<int, int> Parent;
	std::map<int, int> Child;

	std::vector<Eigen::Matrix4f> Local_Coordinate;

	//从数据集中提取出来的
	Eigen::VectorXf Hand_Shape_var;     //shape的方差
	Eigen::MatrixXf Hands_coeffs;       //经过PCA转换矩阵转换后的pose参数（不含手腕）
	Eigen::MatrixXf Hands_components;   //Pose的PCA主成分 45*45 ，从左到右重要性依次递减
	Eigen::VectorXf Hands_mean;
	Eigen::VectorXf Hand_Pose_var;      //Pose的PCA标准差，从左到右依次递减
	Eigen::VectorXf Hands_Pose_Max;
	Eigen::VectorXf Hands_Pose_Min;

	int K_PCAcomponnet;
	Eigen::VectorXf Glove_Difference_MinPCA;
	Eigen::VectorXf Glove_Difference_MaxPCA;
	Eigen::VectorXf Glove_Difference_VARPCA;

	vector<float> vertices_fitting_weight;
private:
	Camera* camera;
	//控制手型的参数
	//这些是控制形状的值
	Eigen::VectorXf Shape_params;
	Eigen::VectorXf Pose_params;   //0-2控制手的全局位置，3-5控制手腕的旋转，6-50控制手势的状态；

	bool want_shapemodel = true;
	bool change_shape = false;


	//这些是计算得到的输出
	Eigen::MatrixXf V_shaped;  //自然状态下，经过形状变换(shape blend)后的顶点
	Eigen::MatrixXf J_shaped;
	Eigen::MatrixXf V_posed;   //自然状态下，经过姿态变换(pose blend 或者 corrective blend shape)之后的顶点


							   //这些都是从文件中读出来的
	Eigen::MatrixXf J;
	Eigen::SparseMatrix<float> J_regressor;
	Eigen::MatrixXf Kintree_table;
	std::vector<Eigen::MatrixXf> Posedirs;
	std::vector<Eigen::MatrixXf> Shapedirs;
	std::vector<Eigen::MatrixXf> Joint_Shapedir;  //这个是根据Joint_regressor和shape_dir算出来的
	Eigen::MatrixXf V_template;
	Eigen::MatrixXf Weights;

	//与变换有关，存储雅各比矩阵
	std::vector<std::vector<Eigen::Matrix4f>> pose_jacobain_matrix;
	std::vector<std::vector<Eigen::Matrix4f>> shape_jacobain_matrix;
	std::vector<std::vector<int>> joint_relation;
public:
	HandModel(Camera *_camera);
	~HandModel() {};

	//更新顶点和关节点的函数
	void UpdataModel();
	void Jacob_Matrix_Updata();
	void set_Shape_Params(const Eigen::VectorXf &shape_params)
	{
		assert(shape_params.size() == this->Shape_params.size());

		for (int i = 0; i < NUM_HAND_SHAPE_PARAMS; ++i) this->Shape_params[i] = shape_params[i];
		this->change_shape = true;
	}
	void set_Pose_Params(const Eigen::VectorXf &pose_params)
	{
		assert(pose_params.size() == this->Pose_params.size());

		for (int i = 0; i < NUM_HAND_POSE_PARAMS; ++i) this->Pose_params[i] = pose_params[i];
	}

	void Save_as_obj();

	//两个雅各比函数
	void Shape_jacobain(Eigen::MatrixXf& jacobain, int vertex_id, const Eigen::Vector3f& vertex_pos = Eigen::Vector3f::Zero());
	void Pose_jacobain(Eigen::MatrixXf& jacobain, int vertex_id, const Eigen::Vector3f& vertex_pos = Eigen::Vector3f::Zero());
	void Joint_Pose_jacobain(Eigen::MatrixXf& jacobain, int joint_id);
private:
	void LoadModel();
	void Load_J(const char* filename);
	void Load_J_regressor(const char* filename);
	void Load_F(const char* filename);
	void Load_Kintree_table(const char* filename);
	void Load_Posedirs(const char* filename);
	void Load_Shapedirs(const char* filename);
	void Load_V_template(const char* filename);
	void Load_Weights(const char* filename);

	void Load_Hands_coeffs(const char* filename);
	void Load_Hands_components(const char* filename);
	void Load_Hands_mean(const char* filename);
	void Load_Hand_Pose_var(const char* filename);
	void Load_Hand_Shape_var(const char* filename);
	void Load_Hand_Pose_Max_Min(const char* filename);

	void Load_GloveDifference_Max_MinPCA(const char* filename);
	void Load_GloveDifference_VarPCA(const char* filename);

	void Load_Vertices_FittingWeight(const char* filename);
	//更新顶点和关节点的函数
	void Updata_V_rest();
	void ShapeSpaceBlend();
	void PoseSpaceBlend();

	void LBS_Updata();
	void NormalUpdata();
	void Pose_Jacobain_matrix_Updata();
	void Shape_Jacobain_matrix_Updata();



	//存放LBSupdata中间值
	std::vector<Eigen::Matrix4f> result;
	std::vector<std::vector<Eigen::Matrix4f>> result_shape_jacob;

	std::vector<Eigen::Matrix4f> result2;
	std::vector<Eigen::Matrix4f> T;

	void Local_Coordinate_Init();
	void Local_Coordinate_Updata();

	void Trans_Matrix_Updata();  //与用于更新后续的四个中间变量，仅在形状改变时候更新
	std::vector<Eigen::Matrix4f> Trans_child_to_parent;
	std::vector<std::vector<Eigen::Matrix4f>> Trans_child_to_parent_Jacob;
	std::vector<Eigen::Matrix4f> Trans_world_to_local;
	std::vector<std::vector<Eigen::Matrix4f>> Trans_world_to_local_Jacob;


private:

	//三个工具函数
	//旋转顺序，相对轴xyz的顺序
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
	std::vector<float> lortmin(const Eigen::VectorXf &finger_pose)
	{
		std::vector<float> result(3 * finger_pose.size());  //实际计算是 9* (full_pose->size()/3)
		Eigen::Matrix3f rotate_mat;

		for (int i = 1; i < this->Joints_num; ++i)
		{
			rotate_mat = EularToRotateMatrix(finger_pose[(i - 1) * 3 + 0], finger_pose[(i - 1) * 3 + 1], finger_pose[(i - 1) * 3 + 2]);
			Eigen::Matrix3f LocalCoordinateRotate = this->Local_Coordinate[i].block(0, 0, 3, 3);

			rotate_mat = LocalCoordinateRotate* rotate_mat*LocalCoordinateRotate.inverse();


			result[(i - 1) * 9 + 0] = rotate_mat(0, 0) - 1; result[(i - 1) * 9 + 1] = rotate_mat(0, 1);     result[(i - 1) * 9 + 2] = rotate_mat(0, 2);
			result[(i - 1) * 9 + 3] = rotate_mat(1, 0);     result[(i - 1) * 9 + 4] = rotate_mat(1, 1) - 1; result[(i - 1) * 9 + 5] = rotate_mat(1, 2);
			result[(i - 1) * 9 + 6] = rotate_mat(2, 0);      result[(i - 1) * 9 + 7] = rotate_mat(2, 1);    result[(i - 1) * 9 + 8] = rotate_mat(2, 2) - 1;

		}

		return result;
	}



public:
	Eigen::Vector3f ComputePalmCenterPosition(Eigen::RowVector3f& palm_center);

	cv::Mat HandModel_depthMap;
	cv::Mat HandModel_binaryMap;
	void GenerateDepthMap();
};



/*
参数说明：
joint[0]  ----- wrist
joint[1~3]  ---------食指
joint[4~6]  -------中指
joint[7~9]  -----小指
joint[10~12]   ------无名指
joint[13~15]   ------大拇指
*/