#include"SolverManager.h"

SolverManager::SolverManager(int start_points, Camera* _camera, Object_type obj_type)
{
	mStartPoints = start_points;

	for (int i = 0; i < mStartPoints; ++i)
	{
		Interacted_Object* tmpInteracted_Object;
		HandModel* temHandmodel = new HandModel(_camera);

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
		Worker* tmpWorker = new Worker(tmpInteracted_Object, temHandmodel, _camera);

		mInteracted_Object.push_back(tmpInteracted_Object);
		mHandModel.push_back(temHandmodel);
		mWorker.push_back(tmpWorker);
	}
}

void SolverManager::Solve(vector<Eigen::VectorXf>& inputData, 
	Image_InputData& imageData, 
	Glove_InputData& gloveData,
	bool previous_success, 
	const Eigen::VectorXf& previous_best_estimation)
{
	std::vector<std::thread> threads;

	for (int i = 0; i < mStartPoints; ++i)
	{
		//这里创建线程的时候注意一定要是传入 mWorker的指针或者引用，不然会创建一个mWorker的临时对象，运算结果会随着临时对象被析构掉
		threads.emplace_back(std::thread(&Worker::Tracking, 
			mWorker[i], 
			inputData[i], 
			imageData, 
			gloveData,
			previous_success, 
			previous_best_estimation));
	}
	for (auto & th : threads)
		th.join();

	//mWorker[0]->Tracking(inputData[0], imageData, gloveData,previous_success, previous_best_estimation);
}


void SolverManager::GetBestEstimation(float& error, 
	bool& is_success, 
	Eigen::VectorXf& params, 
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
		params = mWorker[bestIndex]->Total_Params;
		rendered_images = mWorker[bestIndex]->mRendered_Images;
	}
}