#pragma once
#include"Worker.h"
#include"Interacted_Object.h"
#include<thread>

class SolverManager
{
private:
	int mStartPoints;
	vector<Interacted_Object*> mInteracted_Object;
	vector<Worker*> mWorker;

public:
	SolverManager(int start_points, Camera* _camera,Object_type obj_type);
	void Solve(vector<Eigen::VectorXf>& inputData, Image_InputData& imageData, bool previous_success, const Eigen::VectorXf& previous_best_estimation);
	void GetBestEstimation(float& error, bool& is_success, Eigen::VectorXf& params);
};