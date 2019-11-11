#pragma once
#include"Worker.h"
#include"Interacted_Object.h"
#include"HandModel.h"
#include<thread>

class SolverManager
{
private:
	int mStartPoints;
	vector<Worker*> mWorker;

public:
	SolverManager(int start_points, Camera* _camera,vector<Object_type>& obj_type);
	void Solve(Image_InputData& imageData, Glove_InputData& gloveData,
		vector<Eigen::VectorXf>& hand_init, vector<vector<Eigen::VectorXf>>& object_init,
		bool pre_success,
		Eigen::VectorXf& pre_handPrams, vector<Eigen::VectorXf>& pre_objectParams);
	void GetBestEstimation(float& error, bool& is_success, 
		Eigen::VectorXf& hand_params, vector<Eigen::VectorXf>& obj_params,
		Rendered_Images& rendered_images);
};