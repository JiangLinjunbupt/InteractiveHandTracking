#include"SolverManager.h"

SolverManager::SolverManager(int start_points, Camera* _camera, vector<Object_type>& obj_type)
{
	mStartPoints = start_points;

	for (int i = 0; i < mStartPoints; ++i)
	{
		Worker* tmpWorker = new Worker(_camera, obj_type);
		mWorker.push_back(tmpWorker);
	}
}

void SolverManager::Solve(Image_InputData& imageData, Glove_InputData& gloveData,
	vector<Eigen::VectorXf>& hand_init, vector<vector<Eigen::VectorXf>>& object_init,
	bool pre_success,
	Eigen::VectorXf& pre_handPrams, vector<Eigen::VectorXf>& pre_objectParams)
{
	std::vector<std::thread> threads;

	for (int i = 0; i < mStartPoints; ++i)
	{
		//这里创建线程的时候注意一定要是传入 mWorker的指针或者引用，不然会创建一个mWorker的临时对象，运算结果会随着临时对象被析构掉
		threads.emplace_back(std::thread(&Worker::Tracking, 
			mWorker[i], 
			imageData, gloveData,
			hand_init[i], object_init[i],
			pre_success,
			pre_handPrams, pre_objectParams));
	}
	for (auto & th : threads)
		th.join();

	//mWorker[0]->Tracking(inputData[0], imageData, gloveData,previous_success, previous_best_estimation);
}


void SolverManager::GetBestEstimation(float& error, bool& is_success,
	Eigen::VectorXf& hand_params, vector<Eigen::VectorXf>& obj_params,
	Rendered_Images& rendered_images)
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
	if (bestIndex >= 0)
	{
		hand_params = mWorker[bestIndex]->Hand_Params;
		obj_params = mWorker[bestIndex]->Object_params;
		rendered_images = mWorker[bestIndex]->mRendered_Images;
	}
}