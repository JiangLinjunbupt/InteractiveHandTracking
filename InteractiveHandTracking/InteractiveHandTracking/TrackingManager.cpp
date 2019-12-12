#include"TrackingManager.h"

TrackingManager::TrackingManager(GlobalSetting& setting)
{
	if (setting.start_points > 3)
		setting.start_points = 3;
	mRuntimeType = setting.type;
	mCamera = new Camera(setting.type);
	mInputManager = new InputManager(setting.type, setting.object_type, setting.sharedMeneryPtr, setting.maxPixelNUM);
	mSolverManager = new SolverManager(setting.start_points, mCamera, setting.object_type);

	for (size_t obj_id = 0; obj_id < setting.object_type.size(); ++obj_id)
	{
		Interacted_Object* tmpObject = nullptr;
		switch (setting.object_type[obj_id])
		{
		case yellowSphere:
			tmpObject = new YellowSphere(mCamera);
			break;
		case redCube:
			tmpObject = new RedCube(mCamera);
			break;
		case greenCylinder:
			tmpObject = new GreenCylinder(mCamera);
			break;
		default:
			break;
		}
		mInteracted_Object.push_back(tmpObject);
	}
	mHandModel = new HandModel(mCamera);

	pre_ObjParams.resize(mInteracted_Object.size(), Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS));
	pre_HandParams = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);
	is_success = false;
	mRendered_Images.init(mCamera->width(), mCamera->height());

	fixed_point_belong.clear();
	fixed_contact_Points_local.clear();
	relative_Trans.clear();
	Obj_status_vector.clear();
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

void TrackingManager::GeneratedStartPoints(vector<Eigen::VectorXf>& hand_init, vector<vector<Eigen::VectorXf>>& obj_init)
{
	vector<Eigen::VectorXf> obj_param;
	for (size_t obj_id = 0; obj_id < mInteracted_Object.size(); ++obj_id)
	{
		Eigen::VectorXf tmp = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
		tmp.head(3) = mInputManager->mInputData.image_data.item[obj_id].center;
		obj_param.emplace_back(tmp);
	}

	//起始点3，手套数据的变化结合上一帧数据
	{
		if (is_success)
		{
			Eigen::VectorXf pre_hand = pre_HandParams.tail(NUM_HAND_POSE_PARAMS);
			Eigen::VectorXf pre_glove = mInputManager->mInputData.glove_data.pre_params;
			Eigen::VectorXf now_glove = mInputManager->mInputData.glove_data.params;

			Eigen::VectorXf hand_start3 = pre_hand;
			//分别计算变化
			//其余关节的主要旋转变换
			{
				hand_start3 += (now_glove - pre_glove);
			}
			//手腕的旋转变换
			{
				Eigen::Matrix3f R_hand = (Eigen::AngleAxisf(pre_hand(3), Eigen::Vector3f::UnitX())*
					Eigen::AngleAxisf(pre_hand(4), Eigen::Vector3f::UnitY())*
					Eigen::AngleAxisf(pre_hand(5), Eigen::Vector3f::UnitZ())).toRotationMatrix();

				Eigen::Matrix3f R_pre_glove = (Eigen::AngleAxisf(pre_glove(3), Eigen::Vector3f::UnitX())*
					Eigen::AngleAxisf(pre_glove(4), Eigen::Vector3f::UnitY())*
					Eigen::AngleAxisf(pre_glove(5), Eigen::Vector3f::UnitZ())).toRotationMatrix();

				Eigen::Matrix3f R_now_glove = (Eigen::AngleAxisf(now_glove(3), Eigen::Vector3f::UnitX())*
					Eigen::AngleAxisf(now_glove(4), Eigen::Vector3f::UnitY())*
					Eigen::AngleAxisf(now_glove(5), Eigen::Vector3f::UnitZ())).toRotationMatrix();

				Eigen::VectorXf eular = (R_now_glove * R_pre_glove.inverse() * R_hand).eulerAngles(0, 1, 2);

				hand_start3(3) = eular(0);
				hand_start3(4) = eular(1);
				hand_start3(5) = eular(2);
			}
			//大拇指根部的旋转变换
			{
				Eigen::Matrix3f R_hand = (Eigen::AngleAxisf(pre_hand(42), Eigen::Vector3f::UnitX())*
					Eigen::AngleAxisf(pre_hand(43), Eigen::Vector3f::UnitY())*
					Eigen::AngleAxisf(pre_hand(44), Eigen::Vector3f::UnitZ())).toRotationMatrix();

				Eigen::Matrix3f R_pre_glove = (Eigen::AngleAxisf(pre_glove(42), Eigen::Vector3f::UnitX())*
					Eigen::AngleAxisf(pre_glove(43), Eigen::Vector3f::UnitY())*
					Eigen::AngleAxisf(pre_glove(44), Eigen::Vector3f::UnitZ())).toRotationMatrix();

				Eigen::Matrix3f R_now_glove = (Eigen::AngleAxisf(now_glove(42), Eigen::Vector3f::UnitX())*
					Eigen::AngleAxisf(now_glove(43), Eigen::Vector3f::UnitY())*
					Eigen::AngleAxisf(now_glove(44), Eigen::Vector3f::UnitZ())).toRotationMatrix();

				Eigen::VectorXf eular = (R_now_glove * R_pre_glove.inverse() * R_hand).eulerAngles(0, 1, 2);

				hand_start3(42) = eular(0);
				hand_start3(43) = eular(1);
				hand_start3(44) = eular(2);
			}
			hand_start3.head(3) = pre_hand.head(3);

			hand_init.push_back(hand_start3);
			obj_init.push_back(obj_param);
		}
	}

	//起始点1，手套数据 + 图像的数据
	{
		hand_init.push_back(mInputManager->mInputData.glove_data.params);
		obj_init.push_back(obj_param);
	}

	//起始点2，上一帧结果 + 图像数据
	{
		hand_init.push_back(pre_HandParams.tail(NUM_HAND_POSE_PARAMS));
		obj_init.push_back(obj_param);
	}

	//起始点4，充数的，随便什么吧
	{
		Eigen::VectorXf handparams = Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS);
		handparams.head(NUM_HAND_GLOBAL_PARAMS) = mInputManager->mInputData.glove_data.params.head(NUM_HAND_GLOBAL_PARAMS);
		handparams.tail(NUM_HAND_FINGER_PARAMS) = mHandModel->Hands_mean;
		hand_init.push_back(handparams);
		obj_init.push_back(obj_param);
	}
}
void TrackingManager::ApplyOptimizedParams()
{
	mHandModel->set_Shape_Params(pre_HandParams.head(NUM_HAND_SHAPE_PARAMS));
	mHandModel->set_Pose_Params(pre_HandParams.tail(NUM_HAND_POSE_PARAMS));
	mHandModel->UpdataModel();

	for (int obj_idx = 0; obj_idx < mInteracted_Object.size(); ++obj_idx)
		mInteracted_Object[obj_idx]->Update(pre_ObjParams[obj_idx]);
}
void TrackingManager::ApplyInputParams()
{
	mHandModel->set_Pose_Params(mInputManager->mInputData.glove_data.params);
	mHandModel->UpdataModel();

	for (int obj_idx = 0; obj_idx < mInteracted_Object.size(); ++obj_idx)
	{
		Eigen::VectorXf Object_params = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
		Object_params << mInputManager->mInputData.image_data.item[obj_idx].center(0),
			mInputManager->mInputData.image_data.item[obj_idx].center(1),
			mInputManager->mInputData.image_data.item[obj_idx].center(2), 0, 0, 0;
		mInteracted_Object[obj_idx]->Update(Object_params);
	}
}

void TrackingManager::Tracking(bool do_tracking)
{
	if (FetchInput())
	{
		Obj_StatusJudgement();
		FoundContactPoints();

		/*cout << endl;
		cout << "------------------------------------------" << endl;
		cout << "当前物体的状态如下：" << endl;
		cout << "pre_detect : " << Obj_status_vector[0].pre_Detect << endl;
		cout << "now_detect : " << Obj_status_vector[0].now_Detect << endl;
		cout << "pre_contactWithHand : " << Obj_status_vector[0].pre_ContactWithHand << endl;
		cout << "LossTracking : " << Obj_status_vector[0].lossTracking << endl;
		cout << "----------------------------------------------" << endl;*/
		if (do_tracking)
		{
			vector<Eigen::VectorXf> hand_init;
			vector<vector<Eigen::VectorXf>> obj_init;

			GeneratedStartPoints(hand_init,obj_init);

			mSolverManager->Solve(mInputManager->mInputData.image_data, mInputManager->mInputData.glove_data,
				hand_init, obj_init,
				is_success,
				pre_HandParams, pre_ObjParams,
				fixed_point_belong,
				fixed_contact_Points_local,
				relative_Trans,
				Obj_status_vector);

			float tracking_error;
			mSolverManager->GetBestEstimation(tracking_error, is_success, pre_HandParams, pre_ObjParams,mRendered_Images);

			if (is_success)
			{
				cout << "tracking error is : " << tracking_error << endl;
				ApplyOptimizedParams();
				return;
			}
		}

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

void TrackingManager::FoundContactPoints()
{
	fixed_point_belong.clear();
	fixed_contact_Points_local.clear();
	relative_Trans.clear();

	for (int obj_id = 0; obj_id < mInteracted_Object.size(); ++obj_id)
	{
		//之前物体的变换
		Eigen::MatrixXf Obj_Trans = mInteracted_Object[obj_id]->GetObjectTransMatrix();
		Eigen::MatrixXf Hand_Trans = mHandModel->GetHandModelGlobalTransMatrix((pre_HandParams.tail(NUM_HAND_POSE_PARAMS)).head(NUM_HAND_GLOBAL_PARAMS));

		//相对变换
		relative_Trans.push_back(Hand_Trans.inverse() * Obj_Trans);

		if (mInteracted_Object[obj_id]->mObj_status.pre_Detect 
			&& mInteracted_Object[obj_id]->mObj_status.pre_ContactWithHand
			&& (!mInteracted_Object[obj_id]->mObj_status.now_Detect))
		{
			for (int v_id = 0; v_id < mHandModel->Vertex_num; ++v_id)
			{
				//这里偷个懒，就判断常见交互点与物体是否有交互即可 
				if (mHandModel->contactPoints[v_id] == 1)
				{
					Vector3 p(mHandModel->V_Final(v_id, 0), mHandModel->V_Final(v_id, 1), mHandModel->V_Final(v_id, 2));
					if (mInteracted_Object[obj_id]->SDF(p) < 10.0f)
					{
						Vector3 contactPoint = mInteracted_Object[obj_id]->FindTouchPoint(p);
						Eigen::Vector4f tmp_contactPoint(contactPoint(0), contactPoint(1), contactPoint(2), 1);
						//然后将这个接触点反变换回去
						Eigen::Vector4f local_contactPoint = Obj_Trans.inverse() * tmp_contactPoint;

						fixed_contact_Points_local.emplace_back(make_pair(v_id, local_contactPoint));
						fixed_point_belong.push_back(obj_id);
					}
				}
			}
		}
	}
}