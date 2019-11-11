#include"InputManager.h"

InputManager::InputManager(RuntimeType type, vector<Object_type>& object_type, float* sharedMemeryPtr, int maxPixelNUM)
{
	mRuntimeType = type;
	mCamera = new Camera(type);
	mRealSenseSR300 = new RealSenseSensor(mCamera, maxPixelNUM, object_type);
	mGlove = new Glove(sharedMemeryPtr);

	mInputData.Init(mCamera->width(), mCamera->height(), object_type.size());
	if (mRuntimeType == REALTIME) mRealSenseSR300->start();
}

bool InputManager::fetchInputData()
{
	bool fetchResult = false;

	switch (mRuntimeType)
	{
	case REALTIME:
		fetchResult = mRealSenseSR300->concurrent_fetch_streams(mInputData.image_data);
		mGlove->fetch_RealTime_Data(mInputData);
		break;
	default:
		break;
	}
	ShowImage_input(true, true, false);
	return fetchResult;
}

void InputManager::ShowImage_input(bool show_obj, bool show_hand, bool show_color)
{
	if (show_obj)
	{
		for (int obj_id = 0; obj_id < mInputData.image_data.item.size(); ++obj_id)
		{
			string ss = "第 " + to_string(obj_id) + " 个物体分割结果";
			cv::Mat objectMask;
			cv::flip(mInputData.image_data.item[obj_id].silhouette, objectMask, 0);
			cv::imshow(ss, objectMask);
		}
	}
	
	if (show_hand)
	{
		cv::Mat handMask;
		cv::flip(mInputData.image_data.hand.silhouette, handMask, 0);
		cv::imshow("人手分割", handMask);
	}

	if (show_color)
	{
		cv::Mat color;
		cv::flip(mInputData.image_data.color, color, 0);
		cv::imshow("彩色图", color);
	}

	if (show_obj || show_hand || show_color) cv::waitKey(10);
}