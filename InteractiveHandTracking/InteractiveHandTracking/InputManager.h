#pragma once
#include"RealSenseSR300.h"
using namespace std;

class InputManager
{
public:
	Image_InputData mImage_InputData;
private:
	Camera* mCamera = nullptr;
	RealSenseSensor* mRealSenseSR300 = nullptr;
	RuntimeType mRuntimeType;
public:
	InputManager(RuntimeType type, float* sharedMemeryPtr = nullptr, int maxPixelNUM = 192);
	bool fetchInputData();
	RuntimeType getRuntimeType() {
		return mRuntimeType;
	}

public:
	void ShowImage_input(bool show_obj, bool show_hand, bool show_color);
};

