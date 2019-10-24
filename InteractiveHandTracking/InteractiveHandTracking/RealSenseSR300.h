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

class RealSenseSensor
{
public:
	const int BACK_BUFFER = 1;
	const int FRONT_BUFFER = 0;

	cv::Mat color_array[2];
	cv::Mat depth_array[2];
	cv::Mat silhouette_array[2];

	int *idxs_image_BACK_BUFFER;
	int *idxs_image_FRONT_BUFFER;

	Eigen::RowVector3f palm_center[2];

	pcl::PointCloud<pcl::PointXYZ> handPointCloud[2];
protected:
	bool initialized;
	Camera* camera;
	int currentFrame_idx = 0;
	int MaxPixelNUM = 192;

public:
	RealSenseSensor(Camera* _camera, int maxPixelNUM = 192);
	~RealSenseSensor();
	//bool concurrent_fetch_streams(Image_InputData& inputdata);
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

};