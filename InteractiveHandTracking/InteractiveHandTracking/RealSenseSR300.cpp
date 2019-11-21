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


RealSenseSensor::RealSenseSensor(Camera* _camera, int maxPixelNUM, vector<Object_type>& object_type)
{
	camera = _camera;
	initialized = false;
	MaxPixelNUM = maxPixelNUM;
	mObject_type.assign(object_type.begin(), object_type.end());

	distance_transform.init(_camera->width(), _camera->height());
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
		m_Image_InputData[FRONT_BUFFER].Init(camera->width(), camera->height(),mObject_type.size());
		m_Image_InputData[BACK_BUFFER].Init(camera->width(), camera->height(), mObject_type.size());
	}

	sensor_thread_realsense = std::thread(&RealSenseSensor::run, this);
	sensor_thread_realsense.detach();

	this->initialized = true;
	std::cout << "SensorRealSense Initialization Success ! " << std::endl;

	return true;
}

bool RealSenseSensor::concurrent_fetch_streams(Image_InputData& inputdata)
{
	if (currentFrame_idx > 0)
	{
		std::unique_lock<std::mutex> lock(swap_mutex_realsense);
		condition_realsense.wait(lock, [] {return thread_released_realsense; });
		main_released_realsense = false;

		inputdata = m_Image_InputData[FRONT_BUFFER];

		main_released_realsense = true;
		lock.unlock();
		condition_realsense.notify_all();
		return true;
	}
	return  false;
}

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
		cv::Mat sensor_color = cv::Mat(cv::Size(D_width / downsampling_factor, D_height / downsampling_factor), CV_8UC3, cv::Scalar(0,0,0));

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
					MaskFromRealSense.setTo(0);
					bool is_GetHandSegFromRealSense = GetHandSegFromRealSense(MaskFromRealSense, Handsample->depth, outputData, RightPalmCenter);
					bool is_GetDepthAndColor = GetColorAndDepthImage(sensor_depth, sensor_color, projection, Imagesample->depth, Imagesample->color);
					
					if (is_GetDepthAndColor) {
						SegObjectAndHand(MaskFromRealSense, sensor_color, sensor_depth, is_GetHandSegFromRealSense);

						//给乒乓缓存赋值
						cv::flip(sensor_depth, m_Image_InputData[BACK_BUFFER].depth, -1);
						cv::flip(sensor_color, m_Image_InputData[BACK_BUFFER].color, -1);

						for (int obj_id = 0; obj_id < mObject_type.size(); ++obj_id)
							cv::flip(m_Image_InputData[BACK_BUFFER].item[obj_id].silhouette, m_Image_InputData[BACK_BUFFER].item[obj_id].silhouette, -1);
						cv::flip(m_Image_InputData[BACK_BUFFER].hand.silhouette, m_Image_InputData[BACK_BUFFER].hand.silhouette, -1);

						m_Image_InputData[BACK_BUFFER].silhouette = m_Image_InputData[BACK_BUFFER].hand.silhouette.clone();
						for (int obj_id = 0; obj_id < mObject_type.size(); ++obj_id)
							m_Image_InputData[BACK_BUFFER].silhouette.setTo(255, m_Image_InputData[BACK_BUFFER].item[obj_id].silhouette == 255);

						for (int obj_id = 0; obj_id < mObject_type.size(); ++obj_id)
						{
							m_Image_InputData[BACK_BUFFER].item[obj_id].depth = m_Image_InputData[BACK_BUFFER].depth.clone();
							m_Image_InputData[BACK_BUFFER].item[obj_id].depth.setTo(0, m_Image_InputData[BACK_BUFFER].item[obj_id].silhouette == 0);
						}

						m_Image_InputData[BACK_BUFFER].hand.depth = m_Image_InputData[BACK_BUFFER].depth.clone();
						m_Image_InputData[BACK_BUFFER].hand.depth.setTo(0, m_Image_InputData[BACK_BUFFER].hand.silhouette == 0);


						//点云转换
						DepthToPointCloud(m_Image_InputData[BACK_BUFFER]);

						distance_transform.exec(m_Image_InputData[BACK_BUFFER].silhouette.data, 125);
						std::copy(distance_transform.idxs_image_ptr(), 
							distance_transform.idxs_image_ptr() + m_Image_InputData[BACK_BUFFER].silhouette.cols * m_Image_InputData[BACK_BUFFER].silhouette.rows, 
							m_Image_InputData[BACK_BUFFER].idxs_image);

						//交换数据
						{
							std::unique_lock<std::mutex> lock(swap_mutex_realsense);
							condition_realsense.wait(lock, [] {return main_released_realsense; });
							thread_released_realsense = false;

							m_Image_InputData[FRONT_BUFFER] = m_Image_InputData[BACK_BUFFER];
							currentFrame_idx++;

							thread_released_realsense = true;
							lock.unlock();
							condition_realsense.notify_all();
						}
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
		sync_color_pxc->Release();
	}

	return is_getColorAndDepth;
}

void RealSenseSensor::SegObject(cv::Mat& depth, cv::Mat& hsv)
{
	int object_hmin, object_hmax;
	int object_smin, object_smax;
	int object_vmin, object_vmax;

	int s_Max = 255;
	int v_Max = 255;

	int width = camera->width();
	int height = camera->height();
	for (size_t obj_id = 0; obj_id < mObject_type.size(); ++obj_id)
	{
		//黄色的小球的阈值
		switch (mObject_type[obj_id])
		{
		case yellowSphere:
			object_hmin = 40; object_smin = 100; object_vmin = 0;
			object_hmax = 100; object_smax = 255; object_vmax = 255;
			break;
		case redCube:
			object_hmin = 300; object_smin = 130; object_vmin = 80;
			object_hmax = 360; object_smax = 255; object_vmax = 255;
			break;
		default:
			object_hmin = 0; object_smin = 0; object_vmin = 0;
			object_hmax = 360; object_smax = 255; object_vmax = 255;
			break;
		}

		cv::inRange(hsv,
			cv::Scalar(object_hmin, object_smin / float(s_Max), object_vmin / float(v_Max)),
			cv::Scalar(object_hmax, object_smax / float(s_Max), object_vmax / float(v_Max)),
			m_Image_InputData[BACK_BUFFER].item[obj_id].silhouette);
		////经过一次形态学腐蚀膨胀，去除空洞
		//int Object_DILATION_SIZE = 1;
		//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * Object_DILATION_SIZE + 1, 2 * Object_DILATION_SIZE + 1));
		//cv::dilate(objectMask, objectMask, element);

		if (cv::countNonZero(m_Image_InputData[BACK_BUFFER].item[obj_id].silhouette) > 100)
			m_Image_InputData[BACK_BUFFER].item[obj_id].UpdateStatus(true);
		else
			m_Image_InputData[BACK_BUFFER].item[obj_id].UpdateStatus(false);
	}
}

void RealSenseSensor::SegHand(cv::Mat& depth, cv::Mat& hsv, cv::Mat& HandSegFromRealSense, bool is_handSegFromRealsense)
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
		m_Image_InputData[BACK_BUFFER].hand.UpdateStatus(false);
		return;
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
		m_Image_InputData[BACK_BUFFER].hand.UpdateStatus(false);
		return;
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
	if (contours.size() <= 0) {
		m_Image_InputData[BACK_BUFFER].hand.UpdateStatus(false);
		return;
	}
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
	mask.copyTo(m_Image_InputData[BACK_BUFFER].hand.silhouette);

	if (cv::countNonZero(m_Image_InputData[BACK_BUFFER].hand.silhouette) > 400)
		m_Image_InputData[BACK_BUFFER].hand.UpdateStatus(true);
	else
		m_Image_InputData[BACK_BUFFER].hand.UpdateStatus(false);
}
void RealSenseSensor::SegObjectAndHand(cv::Mat& HandSegFromRealSense, cv::Mat& origin_color, cv::Mat& origin_depth, bool is_handSegFromRealsense)
{
	cv::Mat bgr;  //灰度值归一化
	cv::Mat hsv;  //HSV图像
	origin_color.convertTo(bgr, CV_32FC3, 1.0 / 255, 0); //彩色图像的灰度值归一化,图像大小没有变化，但是类型UINT8变为了FLOAT32位，经过这一步再转换为HSV，才是H ：0-360， S：0-1，V:0-1
	cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);    //颜色空间转换
	cv::Mat depth_for_seg = origin_depth.clone();

	SegObject(depth_for_seg, hsv);
	//物体分割后，需要将物体的深度值再深度图中去掉，并且也要从realsense得到的轮廓中去除该物体
	for (size_t obj_id = 0; obj_id < mObject_type.size(); ++obj_id)
	{
		if (m_Image_InputData[BACK_BUFFER].item[obj_id].now_detect)
		{
			depth_for_seg.setTo(cv::Scalar(BACKGROUND_DEPTH), m_Image_InputData[BACK_BUFFER].item[obj_id].silhouette == 255);
			if (is_handSegFromRealsense)
				HandSegFromRealSense.setTo(0, m_Image_InputData[BACK_BUFFER].item[obj_id].silhouette == 255);
		}
	}
	//然后再进行人手分割
	SegHand(depth_for_seg, hsv, HandSegFromRealSense, is_handSegFromRealsense);
}

void RealSenseSensor::DepthToPointCloud(Image_InputData& image_inputData)
{
	for(int obj_id = 0; obj_id<mObject_type.size();++obj_id)
		image_inputData.item[obj_id].ClearPointcloudAndCenter();
	image_inputData.hand.ClearPointcloudAndCenter();

	int cols = image_inputData.depth.cols;
	int rows = image_inputData.depth.rows;
	Vector3 v0, v1, v2, v3, v4, v5, v6, v7, v8;
	Vector3 vn1, vn2, vn3, vn4, vn5, vn6, vn7, vn8;
	
	//对人手处理
	if (image_inputData.hand.now_detect)
	{
		float Hand_InscribCircleradius = 0; Vector2 Hand_InscribCirclecenter = Vector2::Zero();
		FindInscribedCircle(image_inputData.hand.silhouette, Hand_InscribCircleradius, Hand_InscribCirclecenter);

		//下采样率：
		int DownSampleRate;
		int NonZero = cv::countNonZero(image_inputData.silhouette);
		if (NonZero > MaxPixelNUM)
			DownSampleRate = sqrt(NonZero / MaxPixelNUM);
		else
			DownSampleRate = 1;

		Vector3 hand_center = Vector3::Zero();
		int hand_count = 0;

		for (int row = 1; row<rows - 1; row += DownSampleRate)
			for (int col = 1; col < cols - 1; col += DownSampleRate) {
				//对人手分别处理
				if (image_inputData.hand.silhouette.at<uchar>(row, col) == 255)
				{
					v0 = camera->depth_to_world(col, row, image_inputData.depth.at<ushort>(row, col));

					v1 = camera->depth_to_world(col - 1, row - 1, image_inputData.depth.at<ushort>(row - 1, col - 1));
					v2 = camera->depth_to_world(col + 0, row - 1, image_inputData.depth.at<ushort>(row - 1, col + 0));
					v3 = camera->depth_to_world(col + 1, row - 1, image_inputData.depth.at<ushort>(row - 1, col + 1));
					v4 = camera->depth_to_world(col + 1, row + 0, image_inputData.depth.at<ushort>(row + 0, col + 1));
					v5 = camera->depth_to_world(col + 1, row + 1, image_inputData.depth.at<ushort>(row + 1, col + 1));
					v6 = camera->depth_to_world(col + 0, row + 1, image_inputData.depth.at<ushort>(row + 1, col + 0));
					v7 = camera->depth_to_world(col - 1, row + 1, image_inputData.depth.at<ushort>(row + 1, col - 1));
					v8 = camera->depth_to_world(col - 1, row + 0, image_inputData.depth.at<ushort>(row + 0, col - 1));

					v1 = v1 - v0;
					v2 = v2 - v0;
					v3 = v3 - v0;
					v4 = v4 - v0;
					v5 = v5 - v0;
					v6 = v6 - v0;
					v7 = v7 - v0;
					v8 = v8 - v0;

					vn1 = v1.cross(v2); vn1.normalize();
					vn2 = v2.cross(v3); vn2.normalize();
					vn3 = v3.cross(v4); vn3.normalize();
					vn4 = v4.cross(v5); vn4.normalize();

					vn5 = v5.cross(v6); vn5.normalize();
					vn6 = v6.cross(v7); vn6.normalize();
					vn7 = v7.cross(v8); vn7.normalize();
					vn8 = v8.cross(v1); vn8.normalize();

					vn1 = vn1 + vn2 + vn3 + vn4 + vn5 + vn6 + vn7 + vn8;
					vn1.normalize();
					if (vn1.z() > 0) vn1 = -vn1;

					pcl::PointNormal p;
					p.x = v0.x();
					p.y = v0.y();
					p.z = v0.z();
					p.normal_x = vn1.x();
					p.normal_y = vn1.y();
					p.normal_z = vn1.z();

					image_inputData.hand.pointcloud.points.emplace_back(p);

					float distance = (row - Hand_InscribCirclecenter.y())*(row - Hand_InscribCirclecenter.y()) +
						(col - Hand_InscribCirclecenter.x())*(col - Hand_InscribCirclecenter.x());

					if (distance < Hand_InscribCircleradius*Hand_InscribCircleradius)
					{
						hand_center += v0;
						hand_count++;
					}
				}
			}

		if (hand_count > 0) hand_center /= hand_count;
		image_inputData.hand.center = hand_center;
	}

	//对物体处理
	for (int obj_id = 0; obj_id < mObject_type.size(); ++obj_id)
	{
		if (image_inputData.item[obj_id].now_detect)
		{
			float Obj_InscribCircleradius = 0; Vector2 Obj_InscribCirclecenter = Vector2::Zero();
			FindInscribedCircle(image_inputData.item[obj_id].silhouette, Obj_InscribCircleradius, Obj_InscribCirclecenter);

			int ObjectDownSampleRate = 3;

			Vector3 object_center = Vector3::Zero();;
			int object_count = 0;

			for (int row = 1; row < rows - 1; row += ObjectDownSampleRate)
				for (int col = 1; col < cols - 1; col += ObjectDownSampleRate) {
					//对物体处理
					if (image_inputData.item[obj_id].silhouette.at<uchar>(row, col) != 255)
						continue;

					v0 = camera->depth_to_world(col, row, image_inputData.depth.at<ushort>(row, col));

					v1 = camera->depth_to_world(col - 1, row - 1, image_inputData.depth.at<ushort>(row - 1, col - 1));
					v2 = camera->depth_to_world(col + 0, row - 1, image_inputData.depth.at<ushort>(row - 1, col + 0));
					v3 = camera->depth_to_world(col + 1, row - 1, image_inputData.depth.at<ushort>(row - 1, col + 1));
					v4 = camera->depth_to_world(col + 1, row + 0, image_inputData.depth.at<ushort>(row + 0, col + 1));
					v5 = camera->depth_to_world(col + 1, row + 1, image_inputData.depth.at<ushort>(row + 1, col + 1));
					v6 = camera->depth_to_world(col + 0, row + 1, image_inputData.depth.at<ushort>(row + 1, col + 0));
					v7 = camera->depth_to_world(col - 1, row + 1, image_inputData.depth.at<ushort>(row + 1, col - 1));
					v8 = camera->depth_to_world(col - 1, row + 0, image_inputData.depth.at<ushort>(row + 0, col - 1));

					v1 = v1 - v0;
					v2 = v2 - v0;
					v3 = v3 - v0;
					v4 = v4 - v0;
					v5 = v5 - v0;
					v6 = v6 - v0;
					v7 = v7 - v0;
					v8 = v8 - v0;

					vn1 = v1.cross(v2); vn1.normalize();
					vn2 = v2.cross(v3); vn2.normalize();
					vn3 = v3.cross(v4); vn3.normalize();
					vn4 = v4.cross(v5); vn4.normalize();

					vn5 = v5.cross(v6); vn5.normalize();
					vn6 = v6.cross(v7); vn6.normalize();
					vn7 = v7.cross(v8); vn7.normalize();
					vn8 = v8.cross(v1); vn8.normalize();

					vn1 = vn1 + vn2 + vn3 + vn4 + vn5 + vn6 + vn7 + vn8;
					vn1.normalize();
					if (vn1.z() > 0) vn1 = -vn1;

					pcl::PointNormal p;
					p.x = v0.x();
					p.y = v0.y();
					p.z = v0.z();
					p.normal_x = vn1.x();
					p.normal_y = vn1.y();
					p.normal_z = vn1.z();

					image_inputData.item[obj_id].pointcloud.points.emplace_back(p);

					float distance = (row - Obj_InscribCirclecenter.y())*(row - Obj_InscribCirclecenter.y()) +
						(col - Obj_InscribCirclecenter.x())*(col - Obj_InscribCirclecenter.x());

					if (distance < Obj_InscribCircleradius * Obj_InscribCircleradius)
					{
						object_center += v0;
						object_count++;
					}

				}


			if (object_count > 0) object_center /= object_count;
			image_inputData.item[obj_id].center = object_center;

		}
	}
}

void RealSenseSensor::FindInscribedCircle(cv::Mat& silhouette, float& radius, Vector2& center)
{
	int cols = silhouette.cols;
	int rows = silhouette.rows;

	cv::Moments m = cv::moments(silhouette, true);
	int center_x = m.m10 / m.m00;
	int center_y = m.m01 / m.m00;

	cv::Mat dist_image;
	cv::distanceTransform(silhouette, dist_image, CV_DIST_L2, 3);

	int search_area_min_col = center_x - 30 > 0 ? center_x - 30 : 0;
	int search_area_max_col = center_x + 30 > cols ? cols - 1 : center_x + 30;

	int search_area_min_row = center_y - 30 > 0 ? center_y - 30 : 0;
	int search_area_max_row = center_y + 30 > rows ? rows - 1 : center_y + 30;

	int temp = 0, R = 0, cx = 0, cy = 0;

	for (int row = search_area_min_row; row < search_area_max_row; row++)
	{
		for (int col = search_area_min_col; col < search_area_max_col; col++)
		{
			if (silhouette.at<uchar>(row, col) != 0)
			{
				temp = (int)dist_image.ptr<float>(row)[col];
				if (temp > R)
				{
					R = temp;
					cy = row;
					cx = col;
				}
			}
		}
	}

	radius = R;
	center.x() = cx;
	center.y() = cy;
}

#pragma endregion UntilFunction