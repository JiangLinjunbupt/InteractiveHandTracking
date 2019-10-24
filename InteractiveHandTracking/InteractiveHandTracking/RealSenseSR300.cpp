/*
程序中所用的随机森林来自于github ：https://github.com/edoRemelli/hand-seg-rdf
没有详细的引用说明，这里直接给出url
*/
#include"RealSenseSR300.h"

#include <fertilized/fertilized.h>
#include "fertilized/ndarray.h"
#include "fertilized/global.h"


#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <fstream>
#include <algorithm>

// feature extraction params
# define N_FEAT 8.0
// how many points? n_features = (2*N_FEAT +1)*(2*N_FEAT +1)
# define N_FEAT_PER_NODE 100.0
// how far away should we shoot when computing features - should be proportional to N_FEAT 
# define DELTA 12000.0


// resolution on which process frame
# define SRC_COLS 80
# define SRC_ROWS 60
// htrack's resolution
# define OSRC_COLS 320
# define OSRC_ROWS 240
// depth of 0-depth pixels
# define BACKGROUND_DEPTH 3000.0
// rdf parameters 
# define THRESHOLD 0.7
# define N_THREADS 1
// post processing parameters
# define DILATION_SIZE 9
# define KERNEL_SIZE 3
# define GET_CLOSER_TO_SENSOR 700


using namespace fertilized;
using namespace std;
using namespace std::chrono;

#include<thread>
#include<mutex>
#include<condition_variable>

// sensor resolution
int D_width = 640;
int D_height = 480;

std::thread sensor_thread_realsense;
std::mutex swap_mutex_realsense;
std::condition_variable condition_realsense;
bool main_released_realsense = true;
bool thread_released_realsense = true;


int sensor_frame_realsense = 0;
int tracker_frame_realsense = 0;


int n_features = (2 * N_FEAT + 1)*(2 * N_FEAT + 1);

cv::Mat src_X;
cv::Mat src_Y;
cv::Mat mask;


auto soil = Soil<float, float, unsigned int, Result_Types::probabilities>();
auto forest = soil.ForestFromFile("E:\\githubProject\\hand-seg-rdf\\examples\\c++\\ff_handsegmentation.ff");


RealSenseSensor::RealSenseSensor(Camera* _camera, int maxPixelNUM)
{
	camera = _camera;
	initialized = false;
	MaxPixelNUM = maxPixelNUM;
}

RealSenseSensor::~RealSenseSensor()
{
	std::cout << "~RealSenseSensor() function called  " << std::endl;
	if (!initialized) return;
}

bool RealSenseSensor::initialize()
{
	if (!initialized)
	{
		currentFrame_idx = 0;
		color_array[FRONT_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_8UC3, cv::Scalar(0, 0, 0));
		color_array[BACK_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_8UC3, cv::Scalar(0, 0, 0));

		depth_array[FRONT_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_16UC1, cv::Scalar(0));
		depth_array[BACK_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_16UC1, cv::Scalar(0));

		silhouette_array[FRONT_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_8UC1, cv::Scalar(0));
		silhouette_array[BACK_BUFFER] = cv::Mat(cv::Size(OSRC_COLS, OSRC_ROWS), CV_8UC1, cv::Scalar(0));

		idxs_image_FRONT_BUFFER = new int[OSRC_COLS * OSRC_ROWS];
		idxs_image_BACK_BUFFER = new int[OSRC_COLS * OSRC_ROWS];

		palm_center[FRONT_BUFFER] = Eigen::RowVector3f::Zero();
		palm_center[BACK_BUFFER] = Eigen::RowVector3f::Zero();

		handPointCloud[FRONT_BUFFER].points.clear();
		handPointCloud[BACK_BUFFER].points.clear();

		handPointCloud[FRONT_BUFFER].points.reserve(OSRC_COLS*OSRC_ROWS);
		handPointCloud[BACK_BUFFER].points.reserve(OSRC_COLS*OSRC_ROWS);

	}

	sensor_thread_realsense = std::thread(&RealSenseSensor::run, this);
	sensor_thread_realsense.detach();
	//this->run();

	this->initialized = true;
	std::cout << "SensorRealSense Initialization Success ! " << std::endl;

	return true;
}


//bool RealSenseSensor::concurrent_fetch_streams(InputData& inputdata)
//{
//	if (currentFrame_idx > 0)
//	{
//		std::unique_lock<std::mutex> lock(swap_mutex_realsense);
//		condition_realsense.wait(lock, [] {return thread_released_realsense; });
//		main_released_realsense = false;
//
//		inputdata.image_data.color = color_array[FRONT_BUFFER].clone();
//		inputdata.image_data.depth = depth_array[FRONT_BUFFER].clone();
//		inputdata.image_data.silhouette = silhouette_array[FRONT_BUFFER].clone();
//		std::copy(idxs_image_FRONT_BUFFER, idxs_image_FRONT_BUFFER + OSRC_COLS * OSRC_ROWS, inputdata.image_data.idxs_image);
//
//		inputdata.image_data.palm_Center = palm_center[FRONT_BUFFER];
//		inputdata.image_data.pointcloud.points.assign(handPointCloud[FRONT_BUFFER].points.begin(), handPointCloud[FRONT_BUFFER].points.end());
//
//		main_released_realsense = true;
//		lock.unlock();
//		condition_realsense.notify_all();
//		return true;
//	}
//	return  false;
//}

bool RealSenseSensor::run()
{
	PXCSession *session = PXCSession::CreateInstance();
	PXCSession::ImplDesc desc, desc1;
	memset(&desc, 0, sizeof(desc));
	desc.group = PXCSession::IMPL_GROUP_SENSOR;
	desc.subgroup = PXCSession::IMPL_SUBGROUP_VIDEO_CAPTURE;
	if (session->QueryImpl(&desc, 0, &desc1) < PXC_STATUS_NO_ERROR) return false;

	PXCCapture * capture;
	pxcStatus status = session->CreateImpl<PXCCapture>(&desc1, &capture);
	if (status != PXC_STATUS_NO_ERROR) {
		std::cerr << "FATAL ERROR", "Intel RealSense device not plugged?\n(CreateImpl<PXCCapture> failed)\n";
		exit(0);
	}

	PXCCapture::Device* device;
	device = capture->CreateDevice(0);
	PXCProjection *projection = device->CreateProjection();

	PXCSenseManager *sense_manager = session->CreateSenseManager();
	if (!sense_manager) {
		wprintf_s(L"Unable to create the PXCSenseManager\n");
		return -1;
	}

	PXCHandModule *handAnalyzer = NULL;
	status = sense_manager->EnableHand();
	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_COLOR, D_width, D_height, 30);
	sense_manager->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, D_width, D_height, 30);
	if (status != pxcStatus::PXC_STATUS_NO_ERROR)
	{
		wprintf_s(L"Failed to pair the hand module with I/O");
		return -1;
	}

	handAnalyzer = sense_manager->QueryHand();
	if (handAnalyzer == NULL)
	{
		wprintf_s(L"Failed to pair the hand module with I/O");
		return -1;
	}

	if (sense_manager->Init() >= PXC_STATUS_NO_ERROR) {
		PXCHandData* outputData = handAnalyzer->CreateOutput();
		PXCHandConfiguration* config = handAnalyzer->CreateActiveConfiguration();
		config->DisableAllAlerts();
		config->DisableAllGestures();
		config->SetTrackingMode(PXCHandData::TRACKING_MODE_EXTREMITIES);
		config->EnableSegmentationImage(true);
		config->ApplyChanges();

		// standard downscaling used by htrack
		int downsampling_factor = 2;
		cv::Mat MaskFromRealSense = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_8UC1, cv::Scalar(0));
		cv::Mat sensor_depth = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_16UC1, cv::Scalar(0));

		while (true)
		{
			sense_manager->ReleaseFrame();
			pxcStatus sts = sense_manager->AcquireFrame(true);
			if (sts < PXC_STATUS_NO_ERROR) {
				cout << "sense_manager->AcquireFrame failed\n";
				sense_manager->ReleaseFrame();
				continue;
			}

			const PXCCapture::Sample *Handsample;
			const PXCCapture::Sample *Imagesample;

			Handsample = sense_manager->QueryHandSample();
			Imagesample = sense_manager->QuerySample();

			if (Handsample && Imagesample && Handsample->depth && Imagesample->depth && Imagesample->color) {
				if (outputData) {
					outputData->Update();
					PXCHandData::ExtremityData RightPalmCenter; //记得中心点的2D坐标要的xy要除以2；
					bool is_GetHandSegFromRealSense = GetHandSegFromRealSense(MaskFromRealSense, Handsample->depth, outputData, RightPalmCenter);

					if (is_GetHandSegFromRealSense) {
						cv::imshow("来自realsense的mask", MaskFromRealSense);
					}

					bool is_GetDepthAndColor = GetColorAndDepthImage(sensor_depth, color_array[BACK_BUFFER], projection, Imagesample->depth, Imagesample->color);
					if (is_GetDepthAndColor) {
						cv::imshow("彩色图", color_array[BACK_BUFFER]);

						cv::Mat objecMask = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_8UC1, cv::Scalar(0));
						cv::Mat handMask = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_8UC1, cv::Scalar(0));
						std::pair<bool, bool> segResult;
						segResult = SegObjectAndHand(MaskFromRealSense, color_array[BACK_BUFFER], sensor_depth, is_GetHandSegFromRealSense, objecMask, handMask);

						cv::imshow("物体分割", objecMask);
						cv::imshow("人手分割", handMask);
						cv::waitKey(10);
					}
				}
			}
			cv::waitKey(10);
		}
		sense_manager->ReleaseFrame();
	}
	else
	{
		wprintf_s(L"Init Failed");
		if (sense_manager)
		{
			sense_manager->Close();
			sense_manager->Release();
		}
	}
	cout << "Error Quit !!!\n";
	return -1;
}

bool RealSenseSensor::start()
{
	if (!initialized) this->initialize();

	return true;
}

#pragma region UntilFunction
bool RealSenseSensor::GetHandSegFromRealSense(cv::Mat& mask, PXCImage *depth, PXCHandData *handData, PXCHandData::ExtremityData &RightPalmCenter)
{
	mask.setTo(0);

	PXCImage* image_hand = depth;
	pxcStatus status_tmp = PXC_STATUS_NO_ERROR;
	pxcUID handID;
	pxcI32 numOfHands = handData->QueryNumberOfHands();
	PXCImage::ImageData bdata;

	bool is_success = false;
	if (numOfHands > 0)
	{
		for (int i = 0; i < numOfHands; ++i)
		{
			handData->QueryHandId(PXCHandData::AccessOrderType::ACCESS_ORDER_RIGHT_HANDS, i, handID);
			PXCHandData::IHand* hand;

			if (handData->QueryHandDataById(handID, hand) == PXC_STATUS_NO_ERROR)
			{
				hand->QuerySegmentationImage(image_hand);
				hand->QueryExtremityPoint(PXCHandData::ExtremityType::EXTREMITY_CENTER, RightPalmCenter);
				if (image_hand != nullptr && image_hand->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_Y8, &bdata) == PXC_STATUS_NO_ERROR)
				{
					is_success = true;
					pxcBYTE* row = (pxcBYTE*)bdata.planes[0];
					for (int x = 0, x_sub = 0; x_sub < camera->width(); x += 2, x_sub++)
						for (int y = 0, y_sub = 0; y_sub < camera->height(); y += 2, y_sub++)
						{
							if (row[y * D_width + x] != 0)
							{
								mask.at<uchar>(y_sub, x_sub) = 255;
							}
						}
				}
				//cout << "手中心点：" << RightPalmCenter.pointWorld.x << "  " << RightPalmCenter.pointWorld.y << " "<< RightPalmCenter.pointWorld.z<<endl;
				//cv::circle(mask, cv::Point2f(RightPalmCenter.pointImage.x/2, RightPalmCenter.pointImage.y/2), 5, 0, -1);
				image_hand->ReleaseAccess(&bdata);
			}
		}
	}

	return is_success;
}

bool RealSenseSensor::GetColorAndDepthImage(cv::Mat& depthImg, cv::Mat& colorImg, PXCProjection *projection, PXCImage *depth, PXCImage *color)
{
	bool is_getColorAndDepth = false;
	depthImg.setTo(0);
	colorImg.setTo(cv::Scalar(0, 0, 0));
	PXCImage * sync_color_pxc;
	PXCImage::ImageData depth_buffer;
	PXCImage::ImageData color_buffer;

	sync_color_pxc = projection->CreateColorImageMappedToDepth(depth, color);
	if (sync_color_pxc != nullptr)
	{
		if (sync_color_pxc->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_RGB24, &color_buffer) == PXC_STATUS_NO_ERROR &&
			depth->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_DEPTH, &depth_buffer) == PXC_STATUS_NO_ERROR) {
			unsigned short* data = ((unsigned short *)depth_buffer.planes[0]);
			/// 下采样+中值滤波， 获取的深度图是与正常的图片相比，是左右相反的。
			for (int x = 0, x_sub = 0; x_sub < camera->width(); x += 2, x_sub++)
				for (int y = 0, y_sub = 0; y_sub < camera->height(); y += 2, y_sub++) {
					//彩色
					unsigned char r = color_buffer.planes[0][y * D_width * 3 + x * 3 + 0];
					unsigned char g = color_buffer.planes[0][y * D_width * 3 + x * 3 + 1];
					unsigned char b = color_buffer.planes[0][y * D_width * 3 + x * 3 + 2];
					colorImg.at<cv::Vec3b>(y_sub, x_sub) = cv::Vec3b(r, g, b);
					//深度
					if (x == 0 || y == 0) {
						depthImg.at<unsigned short>(y_sub, x_sub) = (data[y*D_width + x] == 0 ? BACKGROUND_DEPTH : data[y*D_width + x]);
						continue;
					}
					//中值滤波器
					std::vector<int> neighbors = {
						data[(y - 1)*D_width + x - 1],
						data[(y - 1)*D_width + x - 0],
						data[(y - 1)*D_width + x + 1],
						data[(y + 0)*D_width + x - 1],
						data[(y + 0)*D_width + x - 0],
						data[(y + 0)*D_width + x + 1],
						data[(y + 1)*D_width + x - 1],
						data[(y + 1)*D_width + x + 0],
						data[(y + 1)*D_width + x + 1]
					};

					std::sort(neighbors.begin(), neighbors.end());
					depthImg.at<unsigned short>(y_sub, x_sub) = (neighbors[4] == 0 ? BACKGROUND_DEPTH : neighbors[4]);
				}

			is_getColorAndDepth = true;
		}
		//这里要记得释放资源
		sync_color_pxc->ReleaseAccess(&color_buffer);
		depth->ReleaseAccess(&depth_buffer);
	}
	sync_color_pxc->Release();

	return is_getColorAndDepth;
}

bool RealSenseSensor::SegObject(cv::Mat& depth, cv::Mat& hsv, cv::Mat& objectMask)
{
	//黄色的小球的阈值
	int object_hmin = 40, object_smin = 150, object_vmin = 0;
	int object_hmax = 100, object_smax = 255, object_vmax = 255;

	int s_Max = 255;
	int v_Max = 255;

	cv::inRange(hsv,
		cv::Scalar(object_hmin, object_smin / float(s_Max), object_vmin / float(v_Max)),
		cv::Scalar(object_hmax, object_smax / float(s_Max), object_vmax / float(v_Max)),
		objectMask);

	//经过一次形态学腐蚀膨胀，去除空洞
	int Object_DILATION_SIZE = 2;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * Object_DILATION_SIZE + 1, 2 * Object_DILATION_SIZE + 1));
	cv::dilate(objectMask, objectMask, element);

	if (cv::countNonZero(objectMask) > 30) return true;
	else return false;
}

bool RealSenseSensor::SegHand(cv::Mat& depth, cv::Mat& hsv, cv::Mat& HandSegFromRealSense, bool is_handSegFromRealsense, cv::Mat& handMask)
{
	int background_hmin = 0, background_smin = 0, background_vmin = 0;
	int background_hmax = 360, background_smax = 255, background_vmax = 5;

	int glove_hmin = 0, glove_smin = 0, glove_vmin = 0;
	int glove_hmax = 360, glove_smax = 255, glove_vmax = 80;

	int s_Max = 255;
	int v_Max = 255;

	int downsampling_factor = 2;
	// downscaling for processing
	int ds = 4;
	cv::Mat sensor_depth_ds = cv::Mat(cv::Size(D_width / (downsampling_factor*ds), D_height / (downsampling_factor * ds)), CV_16UC1, cv::Scalar(0));
	std::vector<cv::Point> locations;
	vector<vector< cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;


	//先提手套的黑色，以及背景的黑色，并且将这些像素以外的像素深度都设为背景深度
	cv::Mat Background_mask;
	cv::inRange(hsv,
		cv::Scalar(background_hmin, background_smin / float(s_Max), background_vmin / float(v_Max)),
		cv::Scalar(background_hmax, background_smax / float(s_Max), background_vmax / float(v_Max)),
		Background_mask);

	cv::Mat Glove_mask;
	cv::inRange(hsv,
		cv::Scalar(glove_hmin, glove_smin / float(s_Max), glove_vmin / float(v_Max)),
		cv::Scalar(glove_hmax, glove_smax / float(s_Max), glove_vmax / float(v_Max)),
		Glove_mask);
	//经过一次形态学腐蚀膨胀，去除空洞
	Glove_mask.setTo(0, Background_mask != 0);
	int Glove_DILATION_SIZE = 4;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * Glove_DILATION_SIZE + 1, 2 * Glove_DILATION_SIZE + 1));
	cv::dilate(Glove_mask, Glove_mask, element);
	depth.setTo(cv::Scalar(BACKGROUND_DEPTH), Glove_mask != 255);

	//然后进行随机森林分割
	cv::resize(depth, sensor_depth_ds, cv::Size(D_width / (downsampling_factor*ds), D_height / (downsampling_factor * ds)), 0, 0, cv::INTER_NEAREST);

	// prepare vector for fast element acces
	sensor_depth_ds.convertTo(src_X, CV_32F);
	float* ptr = (float*)src_X.data;
	size_t elem_step = src_X.step / sizeof(float);

	// build feature vector 
	locations.clear();

	cv::findNonZero(src_X < GET_CLOSER_TO_SENSOR, locations);   //又是一个高级用法，找到所有深度值小与GET_CLOSER_TO_SENSOR的像素点
	int n_samples = locations.size();

	if (n_samples <= 0)
	{
		cout << "深度值小与GET_CLOSER_TO_SENSOR的像素点为零 -----> continue..." << endl;
		return false;
	}

	// allocat memory for new data
	Array<float, 2, 2> new_data = allocate(n_samples, n_features);
	{
		//这里的用法我没看懂，但是应该是这样的：new_data对应的是n_samples*n_feature这样一个特征输入，每一个sample都对应一个feature；接下来使用多线程填入这些feature
		// Extract the lines serially, since the Array class is not thread-safe (yet)
		std::vector<Array<float, 2, 2>::Reference> lines;
		for (int i = 0; i < n_samples; ++i)
		{
			lines.push_back(new_data[i]);
		}

		for (int j = 0; j < n_samples; j++)
		{
			// depth of current pixel
			//Array<float, 2, 2> line = allocate(1, n_features);
			std::vector<float> features;
			float d = (float)ptr[elem_step*locations[j].y + locations[j].x];
			for (int k = 0; k < (2 * N_FEAT + 1); k++)
			{
				int idx_x = locations[j].x + (int)(DELTA / d) * ((k - N_FEAT) / N_FEAT);
				for (int l = 0; l < (2 * N_FEAT + 1); l++)
				{
					int idx_y = locations[j].y + (int)(DELTA / d) * ((l - N_FEAT) / N_FEAT);
					// read data
					if (idx_x < 0 || idx_x > SRC_COLS || idx_y < 0 || idx_y > SRC_ROWS)
					{
						features.push_back(BACKGROUND_DEPTH - d);
						continue;
					}
					float d_idx = (float)ptr[elem_step*idx_y + idx_x];
					features.push_back(d_idx - d);
				}
			}
			std::copy(features.begin(), features.end(), lines[j].getData());
		}
	}

	// predict data
	Array<double, 2, 2> predictions = forest->predict(new_data, N_THREADS);

	// build probability maps for current frame
	// hand
	cv::Mat probabilityMap = cv::Mat::zeros(SRC_ROWS, SRC_COLS, CV_32F);
	for (size_t j = 0; j < locations.size(); j++)
	{
		probabilityMap.at<float>(locations[j]) = predictions[j][1];
	}

	// COMPUTE AVERAGE DEPTH OF HAND BLOB ON LOW RES IMAGE
	// threshold low res hand probability map to obtain hand mask
	cv::Mat mask_ds = probabilityMap > THRESHOLD;
	// find biggest blob, a.k.a. hand 
	cv::findContours(mask_ds, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	if (contours.size() <= 0)
	{
		cout << "contours  is zero   ---->  continue.." << endl;
		return false;
	}
	int idx = 0, largest_component = 0;
	double max_area = 0;
	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		double area = fabs(cv::contourArea(cv::Mat(contours[idx])));
		if (area > max_area)
		{
			max_area = area;
			largest_component = idx;
		}
	}

	// draw biggest blob
	cv::Mat mask_ds_biggest_blob = cv::Mat::zeros(mask_ds.size(), CV_8U);
	cv::drawContours(mask_ds_biggest_blob, contours, largest_component, cv::Scalar(255), CV_FILLED, 8, hierarchy);   //这里mask_ds_biggest_blob的图像并不是连续的，属于有许多空洞的图，正因为如此，后续才会使用dilate对mask进行膨胀，再结合深度rangemask确定最终人手图像。

																													 // compute average depth
	std::pair<float, int> avg;
	for (int row = 0; row < mask_ds_biggest_blob.rows; ++row)
	{
		for (int col = 0; col < mask_ds_biggest_blob.cols; ++col)
		{
			float depth_wrist = sensor_depth_ds.at<ushort>(row, col);
			if (mask_ds_biggest_blob.at<uchar>(row, col) == 255)
			{
				avg.first += depth_wrist;
				avg.second++;
			}
		}
	}
	ushort depth_hand = (avg.second == 0) ? BACKGROUND_DEPTH : avg.first / avg.second;

	cv::Mat probabilityMap_us;

	// UPSAMPLE USING RESIZE: advantages of joint bilateral upsampling are already exploited 
	cv::resize(probabilityMap, probabilityMap_us, depth.size());
	// BUILD HIGH RESOLUTION MASKS FOR HAND AND WRIST
	cv::Mat mask = probabilityMap_us > THRESHOLD;


	// Extract pixels at depth range on hand only
	ushort depth_range = 100;
	cv::Mat range_mask;
	cv::inRange(depth, depth_hand - depth_range, depth_hand + depth_range, range_mask);

	// POSTPROCESSING: APPLY SOME DILATION and SELECT BIGGEST BLOB
	element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * DILATION_SIZE + 1, 2 * DILATION_SIZE + 1));
	cv::dilate(mask, mask, element);

	// deep copy because find contours modifies original image
	cv::Mat pp;
	mask.copyTo(pp);

	contours.clear();
	hierarchy.clear();
	cv::findContours(pp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	if (contours.size() <= 0) return false;
	idx = 0, largest_component = 0;
	max_area = 0;
	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		double area = fabs(cv::contourArea(cv::Mat(contours[idx])));
		//std::cout << area << std::endl;
		if (area > max_area)
		{
			max_area = area;
			largest_component = idx;
		}
	}
	cv::Mat dst = cv::Mat::zeros(mask.size(), CV_8U);
	cv::drawContours(dst, contours, largest_component, cv::Scalar(255), CV_FILLED, 8, hierarchy);
	dst.setTo(cv::Scalar(0), range_mask == 0);
	mask.setTo(cv::Scalar(0), dst == 0);

	//最后融合来自realsense的人手分割图
	if (is_handSegFromRealsense) mask.setTo(255, HandSegFromRealSense == 255);
	mask.copyTo(handMask);

	return true;
}
std::pair<bool, bool> RealSenseSensor::SegObjectAndHand(cv::Mat& HandSegFromRealSense, cv::Mat& origin_color, cv::Mat& origin_depth, bool is_handSegFromRealsense, cv::Mat& objectMask, cv::Mat& handMask)
{
	std::pair<bool, bool> segResult = std::pair<bool, bool>(false, false);

	cv::Mat bgr;  //灰度值归一化
	cv::Mat hsv;  //HSV图像
	origin_color.convertTo(bgr, CV_32FC3, 1.0 / 255, 0); //彩色图像的灰度值归一化,图像大小没有变化，但是类型UINT8变为了FLOAT32位，经过这一步再转换为HSV，才是H ：0-360， S：0-1，V:0-1
	cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);    //颜色空间转换
	cv::Mat depth_for_seg = origin_depth.clone();

	segResult.first = SegObject(depth_for_seg, hsv, objectMask);
	//物体分割后，需要将物体的深度值再深度图中去掉，并且也要从realsense得到的轮廓中去除该物体
	depth_for_seg.setTo(cv::Scalar(BACKGROUND_DEPTH), objectMask == 255);
	if (is_handSegFromRealsense) HandSegFromRealSense.setTo(0, objectMask == 255);
	//然后再进行人手分割
	segResult.second = SegHand(depth_for_seg, hsv, HandSegFromRealSense, is_handSegFromRealsense, handMask);

	return segResult;
}
#pragma endregion UntilFunction