#pragma once
#include"Types.h"

class Glove
{
private:
	const Vector3 p1 = Vector3(95.67f, 6.383f, 6.186f);
	const Vector3 p2 = Vector3(7.573f, 1.183f, 26.872f);
	const Vector3 p3 = Vector3(26.883f, -3.557f, -37.023f);
	const float weight1 = 0.42f;
	const float weight2 = 0.29f;
	const float weight3 = 0.29f;
private:
	float* mSharedMemeryPtr = nullptr;
	Vector3 bias;
public:
	Glove(float* sharedMemeryPtr) :mSharedMemeryPtr(sharedMemeryPtr)
	{
		bias = weight1*p1 + weight2*p2 + weight3 * p3 - p1;
	}
	void fetch_RealTime_Data(InputData& inputdata);


private:
	Eigen::Matrix3f EularToRotateMatrix(float x, float y, float z);
	Vector3 ComputebiasWithRotate(float x, float y, float z);
};