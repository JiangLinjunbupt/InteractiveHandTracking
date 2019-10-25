#include"InputManager.h"

InputManager::InputManager(RuntimeType type, float* sharedMemeryPtr, int maxPixelNUM)
{
	mRuntimeType = type;
	mCamera = new Camera(type);
	mRealSenseSR300 = new RealSenseSensor(mCamera, maxPixelNUM);

	mImage_InputData.Init(mCamera->width(), mCamera->height());
	if (mRuntimeType == REALTIME) mRealSenseSR300->start();
}


bool InputManager::fetchInputData()
{
	bool fetchResult = false;

	switch (mRuntimeType)
	{
	case REALTIME:
		fetchResult = mRealSenseSR300->concurrent_fetch_streams(mImage_InputData);
		break;
	default:
		break;
	}

	return fetchResult;
}

void InputManager::ShowImage_input(bool show_obj, bool show_hand, bool show_color)
{
	if (show_obj)
	{
		cv::Mat objectMask;
		cv::flip(mImage_InputData.item.silhouette, objectMask, 0);
		cv::imshow("物体分割", objectMask);
	}
	
	if (show_hand)
	{
		cv::Mat handMask;
		cv::flip(mImage_InputData.hand.silhouette, handMask, 0);
		cv::imshow("人手分割", handMask);
	}

	if (show_color)
	{
		cv::Mat color;
		cv::flip(mImage_InputData.color, color, 0);
		cv::imshow("彩色图", color);
	}

	if (show_obj || show_hand || show_color) cv::waitKey(10);
}