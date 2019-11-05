#pragma once
#include"Interacted_Object.h"
#include"HandModel.h"
#include"Kalman.h"

class Worker
{
private:
	struct Setting {
		float Object_fitting_3D = 1.0f;
		float Object_fitting_2D = 1.0f;

		float Object_Trans_Damping = 1000.0f;
		float Object_Rotate_Damping = 100.0f;

		float Object_Temporal_firstOrder_weight = 2.0f;
		float Object_Temporal_secondOrder_weight = 1.0f;


		float Hand_fitting_3D = 2.0f;
		float Hand_fitting_2D = 1.0f;
		float Hand_Pose_MaxMinLimit_weight = 100000.0f;
		float Hand_Pose_Pcalimit_weight = 200.0f;
		float Hand_Pose_Difference_MAXMIN_PCA_weight = 10000.0f;
		float Hand_Pose_Difference_Var_PCA_weight = 5000;
		float Temporal_finger_params_FirstOrder_weight = 50;
		float Temporal_finger_params_SecondOorder_weight = 5;
		float Hand_Shape_Damping_weight = 5000;
		float Hand_Pose_Damping_weight_For_SmallRotate = 10000;
		float Hand_Pose_Damping_weight_For_BigRotate = 100;
		float Collision_weight = 100.0f;

		int max_itr = 6;
		int max_rigid_itr = 0;
		int frames_interval_between_measurements = 60;

		float tracking_fail_threshold = 70.0f;
	}_settings;

public:
	Setting* const setting = &_settings;
	float tracking_error = 0.0f;
	bool tracking_success = false;

private:
	Camera* mCamera;
	Kalman* kalman;

	int itr = 0;
	int total_itr = 0;
	bool Has_Glove = false;;

	std::queue<Eigen::VectorXf> temporal_Object_params;
	std::queue<Eigen::VectorXf> temporal_finger_params;

	Image_InputData* mImage_InputData;
	Eigen::VectorXf Glove_params;
public:
	HandModel* mHandModel;
	Interacted_Object* mInteracted_Object;

	Rendered_Images mRendered_Images;

	Eigen::VectorXf Total_Params;
	Eigen::VectorXf Object_params;
	Eigen::VectorXf Hand_Params;

	std::vector<DataAndCorrespond> Object_correspond;
	std::vector<DataAndCorrespond> Hand_correspond;
	int num_Object_matched_correspond = 0;
	int num_Hand_matched_correspond = 0;

	Worker(Interacted_Object* _interacted_Object, HandModel* _handmodel, Camera* _camera);
	void Tracking(const Eigen::VectorXf& startData, 
		Image_InputData& imageData, 
		Glove_InputData& gloveData,
		bool previous_success, 
		const Eigen::VectorXf& previous_best_estimation);

private:
	void SetInputData(const Eigen::VectorXf& startData, 
		Image_InputData& imageData,
		Glove_InputData& gloveData, 
		bool previous_success,
		const Eigen::VectorXf& previous_best_estimation);
	void SetTemporalInfo(bool previous_success, const Eigen::VectorXf& previous_best_estimation);
	void One_tracking();
	void Evaluation();


	void FindObjectCorrespond();
	void FindHandCorrespond();

	float Fitting3D(LinearSystem& linear_system);
	void Fitting2D(LinearSystem& linear_system);
	void MaxMinLimit(LinearSystem& linear_system);
	void PcaLimit(LinearSystem& linear_system);
	void GloveDifferenceMaxMinPCALimit(LinearSystem& linear_system);
	void GloveDifferenceVarPCALimit(LinearSystem& linear_system);
	void TemporalLimit(LinearSystem& linear_system, bool first_order);
	void CollisionLimit(LinearSystem& linear_system);
	void Damping(LinearSystem& linear_system);
	void RigidOnly(LinearSystem& linear_system);
	Eigen::VectorXf Solver(LinearSystem& linear_system);

};