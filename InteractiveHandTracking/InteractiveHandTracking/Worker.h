#pragma once
#include"Interacted_Object.h"

class Worker
{
private:
	struct Setting {
		float Object_fitting_3D = 1.0f;
		float Object_fitting_2D = 1.0f;

		float Object_Trans_Damping = 1000.0f;
		float Object_Rotate_Damping = 100.0f;

		float Object_Temporal_firstOrder_weight = 5.0f;
		float Object_Temporal_secondOrder_weight = 2.0f;

		int max_itr = 6;
		float tracking_fail_threshold = 30.0f;
	}_settings;

public:
	Setting* const setting = &_settings;
	float tracking_error = 0.0f;
	bool tracking_success = false;

private:
	Camera* mCamera;

	int itr = 0;
	int total_itr = 0;

	std::queue< Eigen::VectorXf> temporal_Object_params;
	int Object_params_num;
	Image_InputData* mImage_InputData;

public:
	Interacted_Object* mInteracted_Object;
	Eigen::VectorXf Object_params;
	std::vector<DataAndCorrespond> correspond;
	int num_matched_correspond = 0;

	Worker(Interacted_Object* _interacted_Object, Camera* _camera);
	void Tracking(const Eigen::VectorXf& startData, Image_InputData& imageData, bool previous_success, const Eigen::VectorXf& previous_best_estimation);

private:
	void SetInputData(const Eigen::VectorXf& startData, Image_InputData& imageData);
	void SetTemporalInfo(bool previous_success, const Eigen::VectorXf& previous_best_estimation);
	void One_tracking();
	void Evaluation();


	void FindObjectCorrespond();
	void Fitting3D(LinearSystem& linear_system);
	void Fitting2D(LinearSystem& linear_system);
	void Damping(LinearSystem& linear_system);
	void TemporalLimit(LinearSystem& linear_system, bool first_order);
	Eigen::VectorXf Solver(LinearSystem& linear_system);

};