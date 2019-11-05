#include"TrackingManager.h"

TrackingManager::TrackingManager(const GlobalSetting& setting)
{
	mRuntimeType = setting.type;
	mCamera = new Camera(setting.type);
	mInputManager = new InputManager(setting.type, setting.sharedMeneryPtr, setting.maxPixelNUM,setting.object_type);
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
	mHandModel = new HandModel(mCamera);

	mPreviousOptimizedParams = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);
	is_success = false;
	mRendered_Images.init(mCamera->width(), mCamera->height());
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
	//起始点1，手套+图像初始化
	{
		Eigen::VectorXf start_point1 = Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);
		start_point1.head(NUM_HAND_POSE_PARAMS) = mInputManager->mInputData.glove_data.params;
		Eigen::VectorXf Object_params = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
		Object_params << mInputManager->mInputData.image_data.item.center(0),
			mInputManager->mInputData.image_data.item.center(1),
			mInputManager->mInputData.image_data.item.center(2), 0, 0, 0;
		start_point1.tail(NUM_OBJECT_PARAMS) = Object_params;
		start_points.push_back(start_point1);
	}

	//起始点2，上一帧的结果
	{
		Eigen::VectorXf start_point2 = mPreviousOptimizedParams.tail(NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);
		Eigen::VectorXf Object_params = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
		Object_params << mInputManager->mInputData.image_data.item.center(0),
			mInputManager->mInputData.image_data.item.center(1),
			mInputManager->mInputData.image_data.item.center(2), 0, 0, 0;
		start_point2.tail(NUM_OBJECT_PARAMS) = Object_params;
		start_points.push_back(start_point2);
	}

	//起始点2，上一帧的结果
	{
		start_points.push_back(mPreviousOptimizedParams.tail(NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS));
	}


	//起始点4,上一帧的xyz和手腕旋转 + 手指的平均数据 + 上一帧物体位置
	{
		Eigen::VectorXf start_point4 = Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);
		start_point4 = mPreviousOptimizedParams.tail(NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);
		(start_point4.head(NUM_HAND_POSE_PARAMS)).tail(NUM_HAND_FINGER_PARAMS) = mHandModel->Hands_mean;
		start_points.push_back(start_point4);
	}

	//起始点5，上一帧手套xyz和手套手腕 + 手指平均位置 + 上一帧物体
	{
		Eigen::VectorXf start_point5 = Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);
		start_point5 = mPreviousOptimizedParams.tail(NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);
		(start_point5.head(NUM_HAND_POSE_PARAMS)).tail(NUM_HAND_FINGER_PARAMS) = mHandModel->Hands_mean;
		Eigen::VectorXf Object_params = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
		Object_params << mInputManager->mInputData.image_data.item.center(0),
			mInputManager->mInputData.image_data.item.center(1),
			mInputManager->mInputData.image_data.item.center(2), 0, 0, 0;
		start_point5.tail(NUM_OBJECT_PARAMS) = Object_params;
		start_points.push_back(start_point5);
	}
}
void TrackingManager::ApplyOptimizedParams(const Eigen::VectorXf& params)
{
	mHandModel->set_Shape_Params(params.head(NUM_HAND_SHAPE_PARAMS));
	mHandModel->set_Pose_Params(params.segment(NUM_HAND_SHAPE_PARAMS, NUM_HAND_POSE_PARAMS));
	mHandModel->UpdataModel();


	mInteracted_Object->Update(params.tail(NUM_OBJECT_PARAMS));
}
void TrackingManager::ApplyInputParams()
{
	mHandModel->set_Pose_Params(mInputManager->mInputData.glove_data.params);
	mHandModel->UpdataModel();

	Eigen::VectorXf Object_params = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
	Object_params << mInputManager->mInputData.image_data.item.center(0),
		mInputManager->mInputData.image_data.item.center(1),
		mInputManager->mInputData.image_data.item.center(2), 0, 0, 0;
	mInteracted_Object->Update(Object_params);
}

void TrackingManager::Tracking(bool do_tracking)
{
	if (FetchInput())
	{
		if (do_tracking)
		{
			vector<Eigen::VectorXf> start_points;
			GeneratedStartPoints(start_points);
			mSolverManager->Solve(start_points, 
				mInputManager->mInputData.image_data, 
				mInputManager->mInputData.glove_data,
				is_success, 
				mPreviousOptimizedParams);

			float tracking_error;
			mSolverManager->GetBestEstimation(tracking_error, is_success, mPreviousOptimizedParams, mRendered_Images);

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
	cv::Mat tmpColor = mInputManager->mInputData.image_data.color.clone();
	cv::flip(tmpColor, tmpColor, 0);
	if (is_success)
	{
		//分别弄人手和物体的
		//人手的
		{
			cv::Mat tmp;
			cv::flip(mRendered_Images.rendered_hand_silhouette, tmp, 0);

			cv::Mat tmp_depth = mRendered_Images.rendered_hand_depth.clone();
			tmp_depth.setTo(0, mRendered_Images.rendered_hand_silhouette == 0);
			double max, min;
			cv::minMaxIdx(tmp_depth, &min, &max);
			//cout << "max  " << max << "  min : " << min << endl;
			cv::Mat normal_map;
			tmp_depth.convertTo(normal_map, CV_8U, 255.0 / (max - min), -min * 255.0 / (max - min));  //灰度拉升
			cv::Mat junheng_Map;
			cv::equalizeHist(normal_map, junheng_Map);   //直方图均衡提高对比度
			cv::Mat color_map;
			cv::applyColorMap(junheng_Map, color_map, cv::COLORMAP_WINTER);  //伪彩色
			color_map.setTo(cv::Scalar(255, 255, 255), mRendered_Images.rendered_hand_silhouette == 0);  //手部以外其他地方置零
			cv::flip(color_map, color_map, 0);

			//根据掩模进行叠加
			color_map.copyTo(tmpColor, tmp);
		}

		//再弄物体的
		{
			cv::Mat tmp;
			cv::flip(mRendered_Images.rendered_object_silhouette, tmp, 0);

			cv::Mat tmp_depth = mRendered_Images.rendered_object_depth.clone();
			tmp_depth.setTo(0, mRendered_Images.rendered_object_silhouette == 0);
			double max, min;
			cv::minMaxIdx(tmp_depth, &min, &max);
			//cout << "max  " << max << "  min : " << min << endl;
			cv::Mat normal_map;
			tmp_depth.convertTo(normal_map, CV_8U, 255.0 / (max - min), -min * 255.0 / (max - min));  //灰度拉升
			cv::Mat junheng_Map;
			cv::equalizeHist(normal_map, junheng_Map);   //直方图均衡提高对比度
			cv::Mat color_map;
			cv::applyColorMap(junheng_Map, color_map, cv::COLORMAP_COOL);  //伪彩色
			color_map.setTo(cv::Scalar(255, 255, 255), mRendered_Images.rendered_object_silhouette == 0);  //手部以外其他地方置零
			cv::flip(color_map, color_map, 0);

			//根据掩模进行叠加
			color_map.copyTo(tmpColor, tmp);
		}
	}

	cv::imshow("ColorAndDepth", tmpColor);
}