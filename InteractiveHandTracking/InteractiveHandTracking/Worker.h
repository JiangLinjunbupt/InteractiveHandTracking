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

		float Object_Trans_Damping = 10.0f;
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
		float Temporal_finger_params_SecondOorder_weight = 50;
		float Hand_Shape_Damping_weight = 5000;
		float Hand_Pose_Damping_weight_For_SmallRotate = 10000;
		float Hand_Pose_Damping_weight_For_BigRotate = 100;
		float Collision_weight = 100.0f;

		//交互
		float Hand_object_collision = 100.0f;
		float Hand_object_contact = 20.0f;
		float fixed_ContactPooint_weight = 20.0f;

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

	Worker(Camera* _camera, vector<Object_type>& object_type);

	void Tracking(Image_InputData& imageData, Glove_InputData& gloveData,
		Eigen::VectorXf& hand_init, vector<Eigen::VectorXf>& object_init,
		bool pre_success,
		Eigen::VectorXf& pre_handPrams, vector<Eigen::VectorXf>& pre_objectParams,
		vector<int>& _fixed_PointsBelong,
		vector<std::pair<int, Eigen::Vector4f>>& _fixed_contact_Points_local,
		vector<Eigen::Matrix4f>& relative_Trans,
		vector<Obj_status>& Obj_status_vector);

private:
	void SetHandInit(Eigen::VectorXf& hand_init, bool pre_success, Eigen::VectorXf& pre_handParams);
	void SetObjectsInit(vector<Eigen::VectorXf>& object_init, bool pre_success, vector<Eigen::VectorXf>& pre_objectParams);

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
	void Object_Fitting_3D(LinearSystem& linear_system, int obj_idx);
	void Object_Fitting_2D(LinearSystem& linear_system, int obj_idx);
	void Object_TemporalLimit(LinearSystem& linear_system, bool first_order, int obj_idx);
	void Object_CollisionLimit(LinearSystem& linear_system, int obj_idx);
	void Object_Damping(LinearSystem& linear_system);

	//交互产生的碰撞等
	void Hand_Object_Collision(LinearSystem& linear_system);
	void Hand_Object_Contact(LinearSystem& linear_system);

	Eigen::VectorXf Solver(LinearSystem& linear_system);
	void Evaluation();

public:
	vector <std::pair<int, Eigen::Vector4f>> fixed_contactPoints_local;
	vector<std::pair<int, Vector3>> fixed_contactPoints;
	vector<int> fixed_contactPointsBelong;

	void FixContactPointsLimited(LinearSystem& linear_system)
	{
		bool is_effect = false;
		int Num_fixed_contactPoints = fixed_contactPoints_local.size();
		for (int obj_id = 0; obj_id < mInteracted_Objects.size(); ++obj_id)
		{
			if ((!mInteracted_Objects[obj_id]->mObj_status.now_Detect) && mInteracted_Objects[obj_id]->mObj_status.pre_ContactWithHand)
				is_effect = true;
		}

		if(is_effect && Num_fixed_contactPoints > 0)
		{
			Eigen::VectorXf e = Eigen::VectorXf::Zero(Num_fixed_contactPoints * 3);
			Eigen::MatrixXf J = Eigen::MatrixXf::Zero(Num_fixed_contactPoints * 3, NUM_HAND_POSE_PARAMS);
			Eigen::MatrixXf pose_jacob;

			for (int i = 0; i < Num_fixed_contactPoints; ++i)
			{
				int v_id = fixed_contactPoints[i].first;

				e(i * 3 + 0) = fixed_contactPoints[i].second(0) - mHandModel->V_Final(v_id, 0);
				e(i * 3 + 1) = fixed_contactPoints[i].second(1) - mHandModel->V_Final(v_id, 1);
				e(i * 3 + 2) = fixed_contactPoints[i].second(2) - mHandModel->V_Final(v_id, 2);

				float distance = sqrt(e(i * 3 + 0)*e(i * 3 + 0) + e(i * 3 + 1)*e(i * 3 + 1) + e(i * 3 + 2)*e(i * 3 + 2));
				float weight = 1.0f / (1.0f + distance);

				e(i * 3 + 0) *= weight;
				e(i * 3 + 1) *= weight;
				e(i * 3 + 2) *= weight;

				mHandModel->Pose_jacobain(pose_jacob, v_id);

				J.block(i * 3, 0, 3, NUM_HAND_POSE_PARAMS) = weight * pose_jacob;
			}

			linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS,
				NUM_HAND_POSE_PARAMS, NUM_HAND_POSE_PARAMS) += setting->fixed_ContactPooint_weight * J.transpose() * J;
			linear_system.rhs.tail(NUM_HAND_POSE_PARAMS) += setting->fixed_ContactPooint_weight*J.transpose()*e;
		}
	}
};