#include"Kalman.h"


Kalman::Kalman(HandModel* handmodel)
{
	//����handmodel��ֻ���ö�Ӧ�ĳ�ʼ���Ĺ���ֵ���Լ�����
	shape_Sigma_init = Eigen::MatrixXf::Identity(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS);
	shape_Sigma_init.diagonal() = handmodel->Hand_Shape_var;

	InvShape_Sigma_init = shape_Sigma_init.inverse();

	estimated_InvShape_Sigma = InvShape_Sigma_init;
	estimated_values = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
	estimated_hessian = estimated_InvShape_Sigma.transpose() * estimated_InvShape_Sigma;

	//���캯���н� measure ��ص���ʱ����Ϊ��
	measured_hessian = Eigen::MatrixXf::Zero(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS);
	measured_values = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
}


void Kalman::Update_estimate(const Eigen::VectorXf& shape_params)
{
	/*�������ͣ�����kalman�ĸ��¹���
	�� = (��_e * ��_m + ��_m * ��_e��/�� ��_e + ��_m��    //���Ц�_e��ʾestimatedֵ�ķ����_m��ʾ����ֵ�ķ���
	��_e = ��_e*��_m/�� ��_e + ��_m��

	��������ʹ�õ�hessian���� hessian�ͷ���Ĺ�ϵ���ڣ� H.inverse() = �� ��  ����ԭ����� ������˹����
	��ˣ�����H.inverse() = ��֮�󣬸��¹����Ϊ��

	�� = (H_m * ��_m + H_e * ��_e��/�� H_e + H_m��
	H_e = H_e + H_m
	*/

	//���ȸ���measure_value
	measured_values = shape_params;

	//Ȼ����¹���ֵ
	estimated_values = (measured_hessian + estimated_hessian).inverse() * (estimated_hessian * estimated_values + measured_hessian * measured_values);

	//�����¹���ֵ�ķ���
	estimated_hessian = estimated_hessian + measured_hessian;

	//����Ŀ������⣬�����https://stackoverflow.com/questions/51681324/matrixbasesqrt-not-working-in-eigen3
	//ͬʱע��ǰ�ᣬ�������estimated_hessian�ǶԽǾ���
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