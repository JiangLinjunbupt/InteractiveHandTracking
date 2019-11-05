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
	//������;����壺shape_Sigma �ǶԽǾ��󣬶Խ����ϵ�ֵ��ÿ����״�����ı�׼�� : ��
	Eigen::MatrixXf shape_Sigma_init;
	//InvShape_Sigma�� shape_Sigma������� Ҳ��һ���ԽǾ��󣬶Խ����ϵ�ֵ��  1/��
	Eigen::MatrixXf InvShape_Sigma_init;

	//�������������hessian����Ķ�Ӧ��ϵ��  Hessian = ��shape_Sigma��ת�� * shape_Sigma���������Ҳ���� ��InvShape_Sigma��ת��*InvShape_Sigma��
	Eigen::MatrixXf estimated_hessian;
	Eigen::MatrixXf estimated_InvShape_Sigma;  //�������Ǽ������״�������Ƕ����ģ����estimated_hessianӦ���ǶԽǾ�������estimated_InvShape_Sigmaֻ��Ҫ���ڶԽ��߿�������
	Eigen::VectorXf estimated_values;
	Eigen::MatrixXf measured_hessian;
	Eigen::VectorXf measured_values;

public:

	Settings* settings = &_settings;

	Kalman(HandModel* handmodel);
	~Kalman() { delete settings; };

	void ReSet()
	{
		//����handmodel��ֻ���ö�Ӧ�ĳ�ʼ���Ĺ���ֵ���Լ�����
		estimated_InvShape_Sigma = InvShape_Sigma_init;
		estimated_values = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
		estimated_hessian = estimated_InvShape_Sigma.transpose() * estimated_InvShape_Sigma;

		//���캯���н� measure ��ص���ʱ����Ϊ��
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