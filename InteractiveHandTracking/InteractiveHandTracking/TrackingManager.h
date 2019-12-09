#pragma once
#include"InputManager.h"
#include"SolverManager.h"

struct GlobalSetting
{
	RuntimeType type;
	int start_points;
	int maxPixelNUM;
	float* sharedMeneryPtr = NULL;

	vector<Object_type> object_type;
};

class TrackingManager
{
public:
	InputManager* mInputManager;
	SolverManager* mSolverManager;
	vector<Interacted_Object*> mInteracted_Object;
	HandModel* mHandModel;
	Camera* mCamera;

	Eigen::VectorXf pre_HandParams;
	vector<Eigen::VectorXf> pre_ObjParams;
	RuntimeType mRuntimeType;

	vector<int> fixed_point_belong;
	vector<std::pair<int, Eigen::Vector4f>> fixed_contact_Points_local;
	vector<Eigen::Matrix4f> relative_Trans;
	vector<Obj_status> Obj_status_vector;
private:
	bool is_success = false;
	Rendered_Images mRendered_Images;
public:
	TrackingManager(GlobalSetting& setting);
	void Tracking(bool do_tracking);

	void ShowInputColorImage();
	void ShowInputDepthImage();
	void ShowInputSilhouette();
	void ShowInputDepthAddColor();
	void ShowRenderAddColor();

private:
	bool FetchInput();
	void GeneratedStartPoints(vector<Eigen::VectorXf>& hand_init, vector<vector<Eigen::VectorXf>>& obj_init);
	void ApplyOptimizedParams();
	void ApplyInputParams();

	void FoundContactPoints();
	void Obj_StatusJudgement()
	{
		Obj_status_vector.clear();
		for (int obj_id = 0; obj_id < mInteracted_Object.size(); ++obj_id)
		{
			Obj_status tmpObj_status;
			tmpObj_status.now_Detect = mInputManager->mInputData.image_data.item[obj_id].now_detect;
			tmpObj_status.pre_Detect = mInputManager->mInputData.image_data.item[obj_id].pre_detect;
			tmpObj_status.loss_LongTime = mInputManager->mInputData.image_data.item[obj_id].loss_LongTime;

			tmpObj_status.pre_ContactWithHand = mInteracted_Object[obj_id]->mObj_status.pre_ContactWithHand;
			tmpObj_status.lossTracking = mInteracted_Object[obj_id]->mObj_status.lossTracking;

			if (tmpObj_status.now_Detect)
			{
				tmpObj_status.first_appear = ((!tmpObj_status.pre_Detect) && tmpObj_status.lossTracking);

				for (int v_id = 0; v_id < mHandModel->Vertex_num; ++v_id)
				{
					Vector3 p(mHandModel->V_Final(v_id, 0), mHandModel->V_Final(v_id, 1), mHandModel->V_Final(v_id, 2));
					if (mInteracted_Object[obj_id]->SDF(p) < 10.0f)
					{
						tmpObj_status.pre_ContactWithHand = true;
						goto L2;
					}
				}

				tmpObj_status.pre_ContactWithHand = false;
			}
			else
				tmpObj_status.first_appear = false;

			L2:
			tmpObj_status.lossTracking = ((tmpObj_status.loss_LongTime) && (!tmpObj_status.pre_ContactWithHand));

			Obj_status_vector.push_back(tmpObj_status);
			mInteracted_Object[obj_id]->mObj_status = tmpObj_status;
		}
	}
};


