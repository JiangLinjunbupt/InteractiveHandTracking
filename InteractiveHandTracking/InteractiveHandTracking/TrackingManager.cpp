#include"TrackingManager.h"

TrackingManager::TrackingManager(const GlobalSetting& setting)
{
	mRuntimeType = setting.type;
	mCamera = new Camera(setting.type);
	mInputManager = new InputManager(setting.type, nullptr, setting.maxPixelNUM,setting.object_type);
	mSolverManager = new SolverManager(setting.start_points, mCamera, setting.object_type);

	switch (setting.object_type)
	{
	case yellowSphere:
		mInteracted_Object = new YellowSphere(mCamera);
		break;
	case redCube:
		mInteracted_Object = new RedCube(mCamera);
		break;
	default:
		break;
	}

	mPreviousOptimizedParams = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
	is_success = false;
}

bool TrackingManager::FetchInput()
{
	bool fetch_success = false;
	if (mInputManager != nullptr)
	{
		fetch_success = mInputManager->fetchInputData();
	}
	return fetch_success;
}

void TrackingManager::GeneratedStartPoints(vector<Eigen::VectorXf>& start_points)
{
	Eigen::VectorXf start_point1 = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
	start_point1 << mInputManager->mImage_InputData.item.center(0),
		mInputManager->mImage_InputData.item.center(1),
		mInputManager->mImage_InputData.item.center(2),
		0, 0, 0;
	start_points.push_back(start_point1);

	Eigen::VectorXf start_point2 = mPreviousOptimizedParams;
	start_points.push_back(start_point2);
	
	Eigen::VectorXf start_point3 = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
	start_point3 << mInputManager->mImage_InputData.item.center(0),
		mInputManager->mImage_InputData.item.center(1),
		mInputManager->mImage_InputData.item.center(2),
		mPreviousOptimizedParams(3), mPreviousOptimizedParams(4), mPreviousOptimizedParams(5);
	start_points.push_back(start_point3);
}
void TrackingManager::ApplyOptimizedParams(const Eigen::VectorXf& params)
{
	mInteracted_Object->Update(params);
}
void TrackingManager::ApplyInputParams()
{
	Eigen::VectorXf inputParams = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
	inputParams << mInputManager->mImage_InputData.item.center(0),
		mInputManager->mImage_InputData.item.center(1),
		mInputManager->mImage_InputData.item.center(2),
		0, 0, 0;
	mInteracted_Object->Update(inputParams);
}

void TrackingManager::Tracking(bool do_tracking)
{
	if (FetchInput())
	{
		if (do_tracking)
		{
			vector<Eigen::VectorXf> start_points;
			GeneratedStartPoints(start_points);
			mSolverManager->Solve(start_points, mInputManager->mImage_InputData, is_success, mPreviousOptimizedParams);

			float tracking_error;
			mSolverManager->GetBestEstimation(tracking_error, is_success, mPreviousOptimizedParams);

			if (is_success)
			{
				cout << "tracking error is : " << tracking_error << endl;
				ApplyOptimizedParams(mPreviousOptimizedParams);
				return;
			}
		}

		mPreviousOptimizedParams.setZero();
		ApplyInputParams();
	}
	else
		cout << "TrackingManager::Tracking---FetchInput--Failed" << endl;
}

void TrackingManager::ShowRenderAddColor()
{
	cv::Mat tmpColor = mInputManager->mImage_InputData.color.clone();
	cv::flip(tmpColor, tmpColor, 0);
	if (is_success)
	{
		mInteracted_Object->GenerateDepthAndSilhouette();

		//复制一份轮廓图作为掩模图
		cv::Mat tmp;
		cv::flip(mInteracted_Object->generatedSilhouette, tmp, 0);

		//对深度图进行伪彩色处理
		cv::Mat tmp_depth = mInteracted_Object->generatedDepth.clone();
		tmp_depth.setTo(0, mInteracted_Object->generatedSilhouette == 0);
		double max, min;
		cv::minMaxIdx(tmp_depth, &min, &max);
		//cout << "max  " << max << "  min : " << min << endl;
		cv::Mat normal_map;
		tmp_depth.convertTo(normal_map, CV_8U, 255.0 / (max - min), -min * 255.0 / (max - min));  //灰度拉升
		cv::Mat junheng_Map;
		cv::equalizeHist(normal_map, junheng_Map);   //直方图均衡提高对比度
		cv::Mat color_map;
		cv::applyColorMap(junheng_Map, color_map, cv::COLORMAP_COOL);  //伪彩色
		color_map.setTo(cv::Scalar(255, 255, 255), mInteracted_Object->generatedSilhouette == 0);  //手部以外其他地方置零
		cv::flip(color_map, color_map, 0);

		//根据掩模进行叠加
		color_map.copyTo(tmpColor, tmp);
	}

	cv::imshow("ColorAndDepth", tmpColor);
}