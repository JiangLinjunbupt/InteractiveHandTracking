#pragma once
#include"Types.h"
#include"HandModel.h"


class Kalman
{
private:
	struct Settings
	{
		float weight = 500;
		float measurement_noise_value = 100.0f;
		float system_noise_value = 0.0f;

		float fited_threshold = 100.0f;
	}_settings;

private:
	//这里解释矩阵含义：shape_Sigma 是对角矩阵，对角线上的值是每个形状参数的标准差 : σ
	Eigen::MatrixXf shape_Sigma_init;
	//InvShape_Sigma是 shape_Sigma的逆矩阵， 也是一个对角矩阵，对角线上的值是  1/σ
	Eigen::MatrixXf InvShape_Sigma_init;

	//上面两个矩阵和hessian矩阵的对应关系是  Hessian = （shape_Sigma的转置 * shape_Sigma）的逆矩阵，也等于 （InvShape_Sigma的转置*InvShape_Sigma）
	Eigen::MatrixXf estimated_hessian;
	Eigen::MatrixXf estimated_InvShape_Sigma;  //这里我是假设的形状参数都是独立的，因此estimated_hessian应该是对角矩阵，这里estimated_InvShape_Sigma只需要等于对角线开方就行
	Eigen::VectorXf estimated_values;
	Eigen::MatrixXf measured_hessian;
	Eigen::VectorXf measured_values;

public:

	Settings* settings = &_settings;

	Kalman(HandModel* handmodel);
	~Kalman() { delete settings; };

	void ReSet()
	{
		//根据handmodel的只设置对应的初始化的估计值，以及方差
		estimated_InvShape_Sigma = InvShape_Sigma_init;
		estimated_values = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
		estimated_hessian = estimated_InvShape_Sigma.transpose() * estimated_InvShape_Sigma;

		//构造函数中将 measure 相关的暂时设置为零
		measured_hessian = Eigen::MatrixXf::Zero(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS);
		measured_values = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
	}
	void ShowHessian()
	{
		std::cout << endl << estimated_hessian << endl;
		std::cout << endl << estimated_values << endl;
	}
	void Set_measured_hessian(LinearSystem& system);
	void Update_estimate(const Eigen::VectorXf& shape_params);
	void track(LinearSystem& system, const Eigen::VectorXf& shape_params);

	bool judgeFitted()
	{
		if (estimated_InvShape_Sigma.diagonal().minCoeff() > settings->fited_threshold)
			return true;
		else
			return false;
	}
};