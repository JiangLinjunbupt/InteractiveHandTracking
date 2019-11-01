#pragma once
#include"InputManager.h"
#include"SolverManager.h"

struct GlobalSetting
{
	RuntimeType type;
	int start_points;
	int maxPixelNUM;

	Object_type object_type;
};

class TrackingManager
{
public:
	InputManager* mInputManager;
	SolverManager* mSolverManager;
	Interacted_Object* mInteracted_Object;
	Camera* mCamera;

	Eigen::VectorXf mPreviousOptimizedParams;
	RuntimeType mRuntimeType;

private:
	bool is_success = false;
public:
	TrackingManager(const GlobalSetting& setting);
	void Tracking(bool do_tracking);

	void ShowInputColorImage();
	void ShowInputDepthImage();
	void ShowInputSilhouette();
	void ShowInputDepthAddColor();
	void ShowRenderAddColor();

private:
	bool FetchInput();
	void GeneratedStartPoints(vector<Eigen::VectorXf>& start_points);
	void ApplyOptimizedParams(const Eigen::VectorXf& params);
	void ApplyInputParams();
};


