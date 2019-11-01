#include"SolverManager.h"

SolverManager::SolverManager(int start_points, Camera* _camera, Object_type obj_type)
{
	mStartPoints = start_points;

	for (int i = 0; i < mStartPoints; ++i)
	{
		Interacted_Object* tmpInteracted_Object;

		switch (obj_type)
		{
		case yellowSphere:
			tmpInteracted_Object = new YellowSphere(_camera);
			break;
		case redCube:
			tmpInteracted_Object = new RedCube(_camera);
			break;
		default:
			tmpInteracted_Object = nullptr;
			break;
		}
		Worker* tmpWorker = new Worker(tmpInteracted_Object, _camera);

		mInteracted_Object.push_back(tmpInteracted_Object);
		mWorker.push_back(tmpWorker);
	}
}

void SolverManager::Solve(vector<Eigen::VectorXf>& inputData, Image_InputData& imageData, bool previous_success, const Eigen::VectorXf& previous_best_estimation)
{
	std::vector<std::thread> threads;

	for (int i = 0; i < mStartPoints; ++i)
	{
		//���ﴴ���̵߳�ʱ��ע��һ��Ҫ�Ǵ��� mWorker��ָ��������ã���Ȼ�ᴴ��һ��mWorker����ʱ������������������ʱ����������
		threads.emplace_back(std::thread(&Worker::Tracking, mWorker[i], inputData[i], imageData, previous_success, previous_best_estimation));
	}
	for (auto & th : threads)
		th.join();

	//mWorker[0]->Tracking(inputData[0], imageData, previous_success, previous_best_estimation);
}


void SolverManager::GetBestEstimation(float& error, bool& is_success, Eigen::VectorXf& params)
{
	float min_error = 100000.0;
	is_success = false;
	int bestIndex = -1;

	for (int i = 0; i < mStartPoints; ++i)
	{
		if (mWorker[i]->tracking_success)
		{
			is_success = true;
			if (mWorker[i]->tracking_error < min_error)
			{
				bestIndex = i;
				min_error = mWorker[i]->tracking_error;
			}
		}
	}

	error = min_error;
	if (bestIndex >= 0) params = mWorker[bestIndex]->Object_params;
	else params = Eigen::VectorXf::Zero(6);
}