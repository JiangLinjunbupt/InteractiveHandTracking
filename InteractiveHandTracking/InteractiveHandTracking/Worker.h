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

		float Object_Trans_Damping = 20.0f;
		float Object_Rotate_Damping = 50.0f;

		float Object_Temporal_firstOrder_weight = 1.0f;
		float Object_Temporal_secondOrder_weight = 0.0f;


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

		//交互
		float Hand_object_collision = 100.0f;

		int max_itr = 15;
		int max_rigid_itr = 1;
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
	bool Has_Glove = false;

	vector<std::queue<Eigen::VectorXf>> temporal_Object_params;
	std::queue<Eigen::VectorXf> temporal_Hand_params;

	Image_InputData* mImage_InputData;
	Eigen::VectorXf Glove_params;
public:
	HandModel* mHandModel;
	Eigen::VectorXf Hand_Params;
	std::vector<DataAndCorrespond> Hand_correspond;
	int num_Hand_matched_correspond = 0;

	vector<Interacted_Object*> mInteracted_Objects;
	vector<Eigen::VectorXf> Object_params;
	vector<std::vector<DataAndCorrespond>> Object_corresponds;
	vector<int> num_Object_matched_correspond;
	
	Rendered_Images mRendered_Images;

	Worker(Camera* _camera,vector<Object_type>& object_type);

	void Tracking(Image_InputData& imageData, Glove_InputData& gloveData,
		Eigen::VectorXf& hand_init, vector<Eigen::VectorXf>& object_init,
		bool pre_success,
		Eigen::VectorXf& pre_handPrams, vector<Eigen::VectorXf>& pre_objectParams);

private:
	void SetHandInit(Eigen::VectorXf& hand_init, bool pre_success, Eigen::VectorXf& pre_handParams);
	void SetObjectsInit(vector<Eigen::VectorXf>& object_init,bool pre_success, vector<Eigen::VectorXf>& pre_objectParams);

	void SetTemporalInfo(bool pre_success,
		Eigen::VectorXf& pre_handPrams, vector<Eigen::VectorXf>& pre_objectParams);

	void FindObjectCorrespond();
	void FindHandCorrespond();

	void Hand_one_tracking();
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

	void Object_one_tracking(int obj_idx);
	void Object_Fitting_3D(LinearSystem& linear_system,int obj_idx);
	void Object_Fitting_2D(LinearSystem& linear_system, int obj_idx);
	void Object_TemporalLimit(LinearSystem& linear_system, bool first_order, int obj_idx);
	void Object_CollisionLimit(LinearSystem& linear_system, int obj_idx);
	void Object_Damping(LinearSystem& linear_system);

	//交互产生的碰撞等
	void Hand_Object_Collision(LinearSystem& linear_system);

	Eigen::VectorXf Solver(LinearSystem& linear_system);
	void Evaluation();

};