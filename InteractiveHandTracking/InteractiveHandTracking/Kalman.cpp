#include"Kalman.h"


Kalman::Kalman(HandModel* handmodel)
{
	//根据handmodel的只设置对应的初始化的估计值，以及方差
	shape_Sigma_init = Eigen::MatrixXf::Identity(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS);
	shape_Sigma_init.diagonal() = handmodel->Hand_Shape_var;

	InvShape_Sigma_init = shape_Sigma_init.inverse();

	estimated_InvShape_Sigma = InvShape_Sigma_init;
	estimated_values = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
	estimated_hessian = estimated_InvShape_Sigma.transpose() * estimated_InvShape_Sigma;

	//构造函数中将 measure 相关的暂时设置为零
	measured_hessian = Eigen::MatrixXf::Zero(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS);
	measured_values = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
}


void Kalman::Update_estimate(const Eigen::VectorXf& shape_params)
{
	/*稍作解释，根据kalman的更新规则：
	μ = (δ_e * μ_m + δ_m * μ_e）/（ δ_e + δ_m）    //其中δ_e表示estimated值的方差，δ_m表示测量值的方差
	δ_e = δ_e*δ_m/（ δ_e + δ_m）

	但是这里使用的hessian矩阵， hessian和方差的关系在于， H.inverse() = δ ，  具体原因详见 拉普拉斯近似
	因此，带入H.inverse() = δ之后，更新规则变为：

	μ = (H_m * μ_m + H_e * μ_e）/（ H_e + H_m）
	H_e = H_e + H_m
	*/

	//首先更新measure_value
	measured_values = shape_params;

	//然后更新估计值
	estimated_values = (measured_hessian + estimated_hessian).inverse() * (estimated_hessian * estimated_values + measured_hessian * measured_values);

	//最后更新估计值的方差
	estimated_hessian = estimated_hessian + measured_hessian;

	//这里的开方问题，详见：https://stackoverflow.com/questions/51681324/matrixbasesqrt-not-working-in-eigen3
	//同时注意前提，我这里的estimated_hessian是对角矩阵
	estimated_InvShape_Sigma = estimated_hessian.cwiseSqrt();

}

void Kalman::Set_measured_hessian(LinearSystem& system)
{
	//std::cout << "Updata Measured Hessian : \n" << system.lhs.block(0, 0, num_shape_params, num_shape_params).diagonal() << endl;

	this->measured_hessian = 1.0f / settings->measurement_noise_value * system.lhs.block(0, 0, NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS);

	for (int i = 0; i < this->measured_hessian.rows(); ++i)
	{
		for (int j = 0; j < this->measured_hessian.cols(); ++j)
		{
			if (i != j)
				this->measured_hessian(i, j) = 0.0f;
		}
	}
}

void Kalman::track(LinearSystem& system, const Eigen::VectorXf& shape_params)
{
	Eigen::MatrixXf LHS;
	Eigen::VectorXf rhs;

	LHS = estimated_hessian;

	rhs = estimated_InvShape_Sigma * (estimated_values - shape_params);

	system.lhs.block(0, 0, NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS) += settings->weight * LHS;
	system.rhs.head(NUM_HAND_SHAPE_PARAMS) += settings->weight * rhs;
}