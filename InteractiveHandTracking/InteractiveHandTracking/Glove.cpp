#include"Glove.h"

void Glove::fetch_RealTime_Data(InputData& inputdata)
{
	if (mSharedMemeryPtr != nullptr)
	{
		Eigen::VectorXf Rotate_pos = Eigen::VectorXf::Zero(NUM_HAND_WRIST_PARAMS + NUM_HAND_FINGER_PARAMS);

		for (int i = 0; i < (NUM_HAND_WRIST_PARAMS + NUM_HAND_FINGER_PARAMS); ++i)
		{
			Rotate_pos[i] = mSharedMemeryPtr[i] * ANGLE_TO_RADIUS;
		}
		inputdata.glove_data.params.tail(NUM_HAND_WRIST_PARAMS + NUM_HAND_FINGER_PARAMS) = Rotate_pos;
		//TODO : 人手的3D坐标
		inputdata.glove_data.params.head(3) = inputdata.image_data.hand.center - ComputebiasWithRotate(Rotate_pos[0], Rotate_pos[1], Rotate_pos[2]);
	}
	else
	{
		inputdata.glove_data.params.setZero();
		inputdata.glove_data.params.head(3) = inputdata.image_data.hand.center;
	}

}

Eigen::Matrix3f  Glove::EularToRotateMatrix(float x, float y, float z)
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

Vector3  Glove::ComputebiasWithRotate(float x, float y, float z)
{
	Eigen::Matrix4f Rotate_0 = Eigen::Matrix4f::Identity();
	Rotate_0.block(0, 0, 3, 3) = EularToRotateMatrix(x, y, z);

	Eigen::Vector4f temp(bias(0), bias(1), bias(2), 1);
	return (Rotate_0*temp).head(3) + p1;
}