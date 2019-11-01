#pragma once
#include"Types.h"
#include"Camera.h"
#include "pxchandmodule.h"
#include "pxcsensemanager.h"
#include "pxccapture.h"
#include "pxchandconfiguration.h"
#include "pxcvideomodule.h"
#include "pxchanddata.h"
#include "pxcsession.h"
#include "pxcprojection.h"
#include "DistanceTransform.h"
class RealSenseSensor
{
private:
	const int BACK_BUFFER = 1;
	const int FRONT_BUFFER = 0;

	Image_InputData m_Image_InputData[2];

	bool initialized;
	Camera* camera;
	int currentFrame_idx = 0;
	int MaxPixelNUM = 192;
	Object_type mObject_type;
	DistanceTransform distance_transform;

public:
	RealSenseSensor(Camera* _camera, int maxPixelNUM = 192, Object_type object_type= yellowSphere);
	~RealSenseSensor();
	bool concurrent_fetch_streams(Image_InputData& inputdata);
	bool start();

private:
	bool run();
	bool initialize();

private:
	bool GetHandSegFromRealSense(cv::Mat& mask, PXCImage *depth, PXCHandData *handData, PXCHandData::ExtremityData &RightPalmCenter);
	bool GetColorAndDepthImage(cv::Mat& depthImg,cv::Mat& colorImg, PXCProjection *projection, PXCImage *depth, PXCImage *color);

	std::pair<bool, bool> SegObjectAndHand(cv::Mat& HandSegFromRealSense, cv::Mat& origin_color, cv::Mat& origin_depth, bool is_handSegFromRealsense, cv::Mat& objectMask, cv::Mat& handMask);

	bool SegHand(cv::Mat& depth, cv::Mat& hsv, cv::Mat& HandSegFromRealSense, bool is_handSegFromRealsense, cv::Mat& handMask);
	bool SegObject(cv::Mat& depth, cv::Mat& hsv, cv::Mat& objectMask);

	void DepthToPointCloud(Image_InputData& image_inputData);
	void FindInscribedCircle(cv::Mat& silhouette, float& radius, Vector2& center);
};