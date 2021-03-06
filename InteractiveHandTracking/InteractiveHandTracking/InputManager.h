#pragma once
#include"RealSenseSR300.h"
#include"Glove.h"
using namespace std;

class InputManager
{
public:
	InputData mInputData;
private:
	Camera* mCamera = nullptr;
	RealSenseSensor* mRealSenseSR300 = nullptr;
	Glove* mGlove = nullptr;
	RuntimeType mRuntimeType;

	vector<bool> Obj_preDetect;
	vector<bool> Obj_prelossDetect;


public:
	InputManager(RuntimeType type, vector<Object_type>& object_type, float* sharedMemeryPtr = nullptr, int maxPixelNUM = 192);
	bool fetchInputData();
	RuntimeType getRuntimeType() {
		return mRuntimeType;
	}

public:
	void ShowImage_input(bool show_obj, bool show_hand, bool show_color);

private:
	void UpdataObjStatus();
};

