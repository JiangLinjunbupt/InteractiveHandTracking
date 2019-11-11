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
};


