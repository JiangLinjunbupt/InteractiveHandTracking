#include"Worker.h"

Worker::Worker(Interacted_Object* _interacted_Object, HandModel* _handmodel, Camera* _camera) :mInteracted_Object(_interacted_Object), mHandModel(_handmodel),mCamera(_camera)
{
	kalman = new Kalman(_handmodel);
	mRendered_Images.init(mCamera->width(), mCamera->height());

	tracking_error = 0.0f;
	tracking_success = false;
	itr = 0;
	total_itr = 0;

	while (!temporal_Object_params.empty()) temporal_Object_params.pop();
	while (!temporal_finger_params.empty()) temporal_finger_params.pop();

	mImage_InputData = nullptr;
	Glove_params = Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS);
	Hand_Params = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);
	Object_params = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);
	Total_Params = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);

	Hand_correspond.clear();
	Object_correspond.clear();

	mInteracted_Object->Update(Object_params);
	mHandModel->set_Shape_Params(Hand_Params.head(NUM_HAND_SHAPE_PARAMS));
	mHandModel->set_Pose_Params(Hand_Params.tail(NUM_HAND_POSE_PARAMS));
	mHandModel->UpdataModel();
}

void Worker::Tracking(const Eigen::VectorXf& startData, 
	Image_InputData& imageData, 
	Glove_InputData& gloveData,
	bool previous_success, 
	const Eigen::VectorXf& previous_best_estimation)
{
	tracking_success = false;
	itr = 0;

	SetInputData(startData, imageData, gloveData, previous_success, previous_best_estimation);
	SetTemporalInfo(previous_success, previous_best_estimation);

	for (; itr < setting->max_itr; ++itr)
	{
		FindObjectCorrespond();
		FindHandCorrespond();

		One_tracking();
	}

	Evaluation();
}

void Worker::SetInputData(const Eigen::VectorXf& startData,
	Image_InputData& imageData,
	Glove_InputData& gloveData,
	bool previous_success,
	const Eigen::VectorXf& previous_best_estimation)
{
	Has_Glove = true;
	//设置输入数据，再设置模型起始点
	mImage_InputData = &imageData;
	Glove_params = gloveData.params;

	//设置人手
	if (previous_success) Hand_Params.head(NUM_HAND_SHAPE_PARAMS) = previous_best_estimation.head(NUM_HAND_SHAPE_PARAMS);
	else
	{
		Hand_Params.head(NUM_HAND_SHAPE_PARAMS) = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
		kalman->ReSet();
	}
	Hand_Params.tail(NUM_HAND_POSE_PARAMS) = startData.head(NUM_HAND_POSE_PARAMS);
	mHandModel->set_Pose_Params(Hand_Params.tail(NUM_HAND_POSE_PARAMS));
	mHandModel->set_Shape_Params(Hand_Params.head(NUM_HAND_SHAPE_PARAMS));
	mHandModel->UpdataModel();

	//设置物体
	Object_params = startData.tail(NUM_OBJECT_PARAMS);
	mInteracted_Object->Update(Object_params);

	//最后组成总的参数
	Total_Params.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS) = Hand_Params;
	Total_Params.tail(NUM_OBJECT_PARAMS) = Object_params;
}

void Worker::SetTemporalInfo(bool previous_success, const Eigen::VectorXf& previous_best_estimation)
{
	if (previous_success)
	{
		if (temporal_Object_params.size() == 2) {
			temporal_Object_params.pop();
			temporal_Object_params.push(previous_best_estimation.tail(NUM_OBJECT_PARAMS));
		}
		else
		{
			temporal_Object_params.push(previous_best_estimation.tail(NUM_OBJECT_PARAMS));
		}

		if (temporal_finger_params.size() == 2)
		{
			temporal_finger_params.pop();
			temporal_finger_params.push(previous_best_estimation.segment(NUM_HAND_SHAPE_PARAMS,NUM_HAND_POSE_PARAMS));
		}
		else
		{
			temporal_finger_params.push(previous_best_estimation.segment(NUM_HAND_SHAPE_PARAMS, NUM_HAND_POSE_PARAMS));
		}
	}
	else
	{
		while (!temporal_Object_params.empty())
			temporal_Object_params.pop();
		while (!temporal_finger_params.empty())
			temporal_finger_params.pop();
	}
}

void Worker::FindObjectCorrespond()
{
	Object_correspond.clear();

	pcl::PointCloud<pcl::PointNormal> object_visible_cloud;
	std::vector<int> visible_idx;
	int Maxsize = mInteracted_Object->Final_Vertices.size();

	object_visible_cloud.reserve(Maxsize);
	visible_idx.reserve(Maxsize);

	for (int i = 0; i < Maxsize; ++i)
	{
		if (mInteracted_Object->Final_Normal[i].z() < 0)
		{
			pcl::PointNormal p;
			p.x = mInteracted_Object->Final_Vertices[i](0);
			p.y = mInteracted_Object->Final_Vertices[i](1);
			p.z = mInteracted_Object->Final_Vertices[i](2);

			p.normal_x = mInteracted_Object->Final_Normal[i](0);
			p.normal_y = mInteracted_Object->Final_Normal[i](1);
			p.normal_z = mInteracted_Object->Final_Normal[i](2);

			object_visible_cloud.points.push_back(p);
			visible_idx.push_back(i);
		}
	}

	//然后再找对应点
	int Numvisible = object_visible_cloud.size();
	int NumPointCloud_sensor = mImage_InputData->item.pointcloud.points.size();

	if (Numvisible > 0 && NumPointCloud_sensor > 0)
	{
		Object_correspond.resize(NumPointCloud_sensor);
		pcl::KdTreeFLANN<pcl::PointNormal> search_tree;
		search_tree.setInputCloud(object_visible_cloud.makeShared());

		const int k = 1;
		std::vector<int> k_indices(k);
		std::vector<float> k_squared_distance(k);

		for (int i = 0; i < NumPointCloud_sensor; ++i)
		{
			search_tree.nearestKSearch(mImage_InputData->item.pointcloud, i, k, k_indices, k_squared_distance);

			Eigen::Vector3f p = Eigen::Vector3f(mImage_InputData->item.pointcloud.points[i].x,
				mImage_InputData->item.pointcloud.points[i].y,
				mImage_InputData->item.pointcloud.points[i].z);
			Object_correspond[i].pointcloud = p;
			

			Eigen::Vector3f p_cor = Eigen::Vector3f(object_visible_cloud.points[k_indices[0]].x,
				object_visible_cloud.points[k_indices[0]].y,
				object_visible_cloud.points[k_indices[0]].z);

			Object_correspond[i].correspond = p_cor;
			Object_correspond[i].correspond_idx = visible_idx[k_indices[0]];

			float distance = (p_cor - p).norm();

			if (distance > 50)
				Object_correspond[i].is_match = false;
			else
				Object_correspond[i].is_match = true;
		}
	}

	num_Object_matched_correspond = 0;

	for (std::vector<DataAndCorrespond>::iterator itr = Object_correspond.begin(); itr != Object_correspond.end(); ++itr)
	{
		if (itr->is_match)
			++num_Object_matched_correspond;
	}
}

void Worker::FindHandCorrespond()
{
	Hand_correspond.clear();
	//首先加载handmodel中的visible point
	pcl::PointCloud<pcl::PointNormal> Handmodel_visible_cloud;
	std::vector<int> visible_idx;

	for (int i = 0; i < mHandModel->Vertex_num; ++i)
	{
		if (mHandModel->V_Normal_Final(i, 2) <= 0)
		{
			pcl::PointNormal p;
			p.x = mHandModel->V_Final(i, 0);
			p.y = mHandModel->V_Final(i, 1);
			p.z = mHandModel->V_Final(i, 2);

			p.normal_x = mHandModel->V_Normal_Final(i, 0);
			p.normal_y = mHandModel->V_Normal_Final(i, 1);
			p.normal_z = mHandModel->V_Normal_Final(i, 2);

			Handmodel_visible_cloud.points.push_back(p);
			visible_idx.push_back(i);
		}
	}


	//然后再找对应点
	int NumVisible_ = Handmodel_visible_cloud.points.size();
	int NumPointCloud_sensor = mImage_InputData->hand.pointcloud.points.size();

	if (NumVisible_ > 0 && NumPointCloud_sensor > 0)
	{
		Hand_correspond.resize(NumPointCloud_sensor);

		pcl::KdTreeFLANN<pcl::PointNormal> search_kdtree;
		search_kdtree.setInputCloud(Handmodel_visible_cloud.makeShared());  //这里注意PCL的flann会和opencv的flann冲突，注意解决

		const int k = 1;
		std::vector<int> k_indices(k);
		std::vector<float> k_squared_distances(k);
		for (int i = 0; i < NumPointCloud_sensor; ++i)
		{
			search_kdtree.nearestKSearch(mImage_InputData->hand.pointcloud, i, k, k_indices, k_squared_distances);

			Eigen::Vector3f p = Eigen::Vector3f(mImage_InputData->hand.pointcloud.points[i].x,
				mImage_InputData->hand.pointcloud.points[i].y,
				mImage_InputData->hand.pointcloud.points[i].z);
			Hand_correspond[i].pointcloud = p;

			Eigen::Vector3f p_2 = Eigen::Vector3f(Handmodel_visible_cloud.points[k_indices[0]].x,
				Handmodel_visible_cloud.points[k_indices[0]].y,
				Handmodel_visible_cloud.points[k_indices[0]].z);

			Hand_correspond[i].correspond = p_2;
			Hand_correspond[i].correspond_idx = visible_idx[k_indices[0]];

			float distance = (p_2 - p).norm();

			if (distance > 30)
				Hand_correspond[i].is_match = false;
			else
				Hand_correspond[i].is_match = true;
		}
	}


	num_Hand_matched_correspond = 0;
	for (int i = 0; i < NumPointCloud_sensor; ++i)
	{
		if (Hand_correspond[i].is_match)
			++num_Hand_matched_correspond;
	}
}

void Worker::One_tracking()
{
	//初始化求解相关
	LinearSystem linear_system;
	linear_system.lhs = Eigen::MatrixXf::Zero(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS, 
		NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);
	linear_system.rhs = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS);


	float error_3D = this->Fitting3D(linear_system);
	this->Fitting2D(linear_system);

	//这里可以开始更新kalman的海森矩阵
	if (itr>(setting->max_itr - 2)
		&& error_3D < 8.0f
		&& !kalman->judgeFitted())  //尽量在更新的迭代后几次，并且跟踪成功的时候更新
		kalman->Set_measured_hessian(linear_system);

	this->MaxMinLimit(linear_system);
	this->PcaLimit(linear_system);
	if (Has_Glove)
	{
		this->GloveDifferenceMaxMinPCALimit(linear_system);
		this->GloveDifferenceVarPCALimit(linear_system);
	}
	this->TemporalLimit(linear_system, true);
	this->TemporalLimit(linear_system, false);
	kalman->track(linear_system, Hand_Params.head(NUM_HAND_SHAPE_PARAMS));
	this->CollisionLimit(linear_system);
	this->Damping(linear_system);
	if (itr < setting->max_rigid_itr)
		this->RigidOnly(linear_system);  //这个一定要放最后

	//求解
	Eigen::VectorXf solution = this->Solver(linear_system);

	for (int i = 0; i < (NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS + NUM_OBJECT_PARAMS); ++i)
		Total_Params[i] += solution[i];

	//这里可以通过total_itr迭代次数判断，通过kalman更新形状参数的间隔
	if (total_itr > 2 * setting->frames_interval_between_measurements
		&& (total_itr % setting->frames_interval_between_measurements) == 0
		&& error_3D < 8.0f
		&& !kalman->judgeFitted())
		kalman->Update_estimate(Total_Params.head(NUM_HAND_SHAPE_PARAMS));


	//更新参数
	Hand_Params = Total_Params.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);
	mHandModel->set_Shape_Params(Hand_Params.head(NUM_HAND_SHAPE_PARAMS));
	mHandModel->set_Pose_Params(Hand_Params.tail(NUM_HAND_POSE_PARAMS));
	mHandModel->UpdataModel();

	Object_params = Total_Params.tail(NUM_OBJECT_PARAMS);
	mInteracted_Object->Update(Object_params);

	total_itr++;
}

float Worker::Fitting3D(LinearSystem& linear_system)
{
	float hand_3D_error = 0.0f;

	//先弄人手的
	{
		int NumofCorrespond = num_Hand_matched_correspond;
		Eigen::VectorXf e = Eigen::VectorXf::Zero(NumofCorrespond * 3);
		Eigen::MatrixXf J = Eigen::MatrixXf::Zero(NumofCorrespond * 3, NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);

		Eigen::MatrixXf shape_jacob, pose_jacob;
		int count = 0;
		int Hand_correspond_size = Hand_correspond.size();
		for (int i = 0; i < Hand_correspond_size; ++i)
		{
			if (Hand_correspond[i].is_match)
			{
				int v_id = Hand_correspond[i].correspond_idx;

				e(count * 3 + 0) = Hand_correspond[i].pointcloud(0) - Hand_correspond[i].correspond(0);
				e(count * 3 + 1) = Hand_correspond[i].pointcloud(1) - Hand_correspond[i].correspond(1);
				e(count * 3 + 2) = Hand_correspond[i].pointcloud(2) - Hand_correspond[i].correspond(2);

				float e_sqrt = sqrt(pow(e(count * 3 + 0), 2) + pow(e(count * 3 + 1), 2) + pow(e(count * 3 + 2), 2));
				hand_3D_error += e_sqrt;
				//这里使用的是Reweighted Least squard error
				//参考：
				//https://www.cs.bgu.ac.il/~mcv172/wiki.files/Lec5.pdf （主要）
				//https://blog.csdn.net/baidu_17640849/article/details/71155537  （辅助）
				//weight = s/(e^2 + s^2)


				//s越大，对异常值的容忍越大
				float s = 50;
				float weight = 1;
				weight = (itr + 1) * s / (e_sqrt + s);

				weight = 6.5f / sqrt(e_sqrt + 0.001);

				e(count * 3 + 0) *= weight;
				e(count * 3 + 1) *= weight;
				e(count * 3 + 2) *= weight;

				mHandModel->Shape_jacobain(shape_jacob, v_id);
				mHandModel->Pose_jacobain(pose_jacob, v_id);

				J.block(count * 3, 0, 3, NUM_HAND_SHAPE_PARAMS) = shape_jacob;
				J.block(count * 3, NUM_HAND_SHAPE_PARAMS, 3, NUM_HAND_POSE_PARAMS) = weight*pose_jacob;

				++count;
			}
		}

		hand_3D_error = hand_3D_error / count;

		Eigen::MatrixXf JtJ = J.transpose()*J;
		Eigen::VectorXf JTe = J.transpose()*e;

		linear_system.lhs.block(0,0,
			NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS, 
			NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS) += setting->Hand_fitting_3D*JtJ;
		linear_system.rhs.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS) += setting->Hand_fitting_3D*JTe;
	}

	//再弄物体的
	{
		int NumofCorrespond = num_Object_matched_correspond;
		Eigen::VectorXf e = Eigen::VectorXf::Zero(NumofCorrespond * 3);
		Eigen::MatrixXf J = Eigen::MatrixXf::Zero(NumofCorrespond * 3, NUM_OBJECT_PARAMS);

		//临时变量
		Eigen::MatrixXf tmp_jacob;
		int count = 0;
		int Object_correspond_size = Object_correspond.size();

		for (int i = 0; i < Object_correspond_size; ++i)
		{
			if (Object_correspond[i].is_match)
			{
				int v_id = Object_correspond[i].correspond_idx;

				e(count * 3 + 0) = Object_correspond[i].pointcloud(0) - Object_correspond[i].correspond(0);
				e(count * 3 + 1) = Object_correspond[i].pointcloud(1) - Object_correspond[i].correspond(1);
				e(count * 3 + 2) = Object_correspond[i].pointcloud(2) - Object_correspond[i].correspond(2);

				float e_sqrt = sqrt(pow(e(count * 3 + 0), 2) + pow(e(count * 3 + 1), 2) + pow(e(count * 3 + 2), 2));
				//这里使用的是Reweighted Least squard error
				//参考：
				//https://www.cs.bgu.ac.il/~mcv172/wiki.files/Lec5.pdf （主要）
				//https://blog.csdn.net/baidu_17640849/article/details/71155537  （辅助）
				//weight = s/(e^2 + s^2)


				//s越大，对异常值的容忍越大
				float s = 50;
				float weight = 1;
				weight = (itr + 1) * s / (e_sqrt + s);

				weight = 6.5f / sqrt(e_sqrt + 0.001);

				e(count * 3 + 0) *= weight;
				e(count * 3 + 1) *= weight;
				e(count * 3 + 2) *= weight;

				mInteracted_Object->object_jacobain(tmp_jacob, v_id);
				J.block(count * 3, 0, 3, NUM_OBJECT_PARAMS) = tmp_jacob;
				++count;
			}
		}

		Eigen::MatrixXf JtJ = J.transpose()*J;
		Eigen::VectorXf JTe = J.transpose()*e;

		linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
			NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
			NUM_OBJECT_PARAMS,NUM_OBJECT_PARAMS) += setting->Object_fitting_3D*JtJ;
		linear_system.rhs.tail(NUM_OBJECT_PARAMS) += setting->Object_fitting_3D*JTe;
	}

	return hand_3D_error;
}

void Worker::Fitting2D(LinearSystem& linear_system)
{
	int width = mCamera->width();
	int height = mCamera->height();
	//先弄人手的
	{
		vector<pair<Eigen::Matrix2Xf, Eigen::Vector2f>> JacobVector_2D;

		int visiblePointSize = mHandModel->V_Visible_2D.size();

		Eigen::MatrixXf shape_jacob, pose_jacob;

		for (int i = 0; i < visiblePointSize; ++i)
		{
			int idx = mHandModel->V_Visible_2D[i].second;

			Eigen::Vector3f pixel_3D_pos(mHandModel->V_Final(idx, 0), mHandModel->V_Final(idx, 1), mHandModel->V_Final(idx, 2));
			Eigen::Vector2i pixel_2D_pos(mHandModel->V_Visible_2D[i].first);

			if (pixel_2D_pos.x() >= 0 && pixel_2D_pos.x() < width && pixel_2D_pos.y() >= 0 && pixel_2D_pos.y() < height)
			{
				int cloest_idx = mImage_InputData->idxs_image[pixel_2D_pos(1) * width + pixel_2D_pos(0)];
				Eigen::Vector2i pixel_2D_cloest;
				pixel_2D_cloest << cloest_idx%width, cloest_idx / width;

				float closet_distance = (pixel_2D_cloest - pixel_2D_pos).norm();

				if (closet_distance > 0)
				{
					//计算 J 和 e
					pair<Eigen::Matrix2Xf, Eigen::Vector2f> J_and_e;

					//先算e
					Eigen::Vector2f e;
					e(0) = (float)pixel_2D_cloest(0) - (float)pixel_2D_pos(0);
					e(1) = (float)pixel_2D_cloest(1) - (float)pixel_2D_pos(1);

					J_and_e.second = e;

					//再计算J
					Eigen::Matrix<float, 2, 3> J_perspective = mCamera->projection_jacobian(pixel_3D_pos);
					Eigen::MatrixXf J_3D = Eigen::MatrixXf::Zero(3, NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);

					mHandModel->Shape_jacobain(shape_jacob, idx);
					mHandModel->Pose_jacobain(pose_jacob, idx);

					J_3D.block(0, 0, 3, NUM_HAND_SHAPE_PARAMS) = shape_jacob;
					J_3D.block(0, NUM_HAND_SHAPE_PARAMS, 3, NUM_HAND_POSE_PARAMS) = pose_jacob;

					J_and_e.first = J_perspective * J_3D;

					JacobVector_2D.push_back(J_and_e);
				}
			}

		}

		int size = JacobVector_2D.size();

		if (size > 0)
		{
			Eigen::MatrixXf J_2D = Eigen::MatrixXf::Zero(2 * size, NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);
			Eigen::VectorXf e_2D = Eigen::VectorXf::Zero(2 * size);

			for (int i = 0; i < size; ++i)
			{
				J_2D.block(i * 2, 0, 2, NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS) = JacobVector_2D[i].first;
				e_2D.segment(i * 2, 2) = JacobVector_2D[i].second;
			}

			linear_system.lhs.block(0,0,
				NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
				NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS )+= setting->Hand_fitting_2D * J_2D.transpose() * J_2D;
			linear_system.rhs.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS) += setting->Hand_fitting_2D * J_2D.transpose() * e_2D;
		}
	}

	//再弄物体的
	{
		vector<pair<Eigen::Matrix2Xf, Eigen::Vector2f>> JacobVector_2D;

		int visiblePointSize = mInteracted_Object->Visible_2D.size();

		Eigen::MatrixXf tmp_jacob;

		for (int i = 0; i < visiblePointSize; ++i)
		{
			int idx = mInteracted_Object->Visible_2D[i].second;

			Eigen::Vector3f pixel_3D_pos = mInteracted_Object->Final_Vertices[idx];
			Eigen::Vector2i pixel_2D_pos = Eigen::Vector2i(mInteracted_Object->Visible_2D[i].first.x(), mInteracted_Object->Visible_2D[i].first.y());

			if (pixel_2D_pos.x() >= 0 && pixel_2D_pos.x() < width && pixel_2D_pos.y() >= 0 && pixel_2D_pos.y() < height)
			{
				int cloest_idx = mImage_InputData->idxs_image[pixel_2D_pos(1) * width + pixel_2D_pos(0)];
				Eigen::Vector2i pixel_2D_cloest;
				pixel_2D_cloest << cloest_idx%width, cloest_idx / width;

				float closet_distance = (pixel_2D_cloest - pixel_2D_pos).norm();

				if (closet_distance > 0)
				{
					//计算 J 和 e
					pair<Eigen::Matrix2Xf, Eigen::Vector2f> J_and_e;

					//先算e
					Eigen::Vector2f e;
					e(0) = (float)pixel_2D_cloest(0) - (float)pixel_2D_pos(0);
					e(1) = (float)pixel_2D_cloest(1) - (float)pixel_2D_pos(1);

					J_and_e.second = e;

					//再计算J
					Eigen::Matrix<float, 2, 3> J_perspective = mCamera->projection_jacobian(pixel_3D_pos);
					mInteracted_Object->object_jacobain(tmp_jacob, idx);

					J_and_e.first = J_perspective * tmp_jacob;

					JacobVector_2D.push_back(J_and_e);
				}
			}
		}

		int size = JacobVector_2D.size();

		if (size > 0)
		{
			Eigen::MatrixXf J_2D = Eigen::MatrixXf::Zero(2 * size, NUM_OBJECT_PARAMS);
			Eigen::VectorXf e_2D = Eigen::VectorXf::Zero(2 * size);

			for (int i = 0; i < size; ++i)
			{
				J_2D.block(i * 2, 0, 2, NUM_OBJECT_PARAMS) = JacobVector_2D[i].first;
				e_2D.segment(i * 2, 2) = JacobVector_2D[i].second;
			}

			linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
				NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
				NUM_OBJECT_PARAMS, NUM_OBJECT_PARAMS) += setting->Object_fitting_2D * J_2D.transpose() * J_2D;
			linear_system.rhs.tail(NUM_OBJECT_PARAMS) += setting->Object_fitting_2D * J_2D.transpose() * e_2D;
		}
	}
}

void Worker::MaxMinLimit(LinearSystem& linear_system)
{
	Eigen::MatrixXf J_limit = Eigen::MatrixXf::Zero(NUM_HAND_FINGER_PARAMS, NUM_HAND_FINGER_PARAMS);
	Eigen::VectorXf e_limit = Eigen::VectorXf::Zero(NUM_HAND_FINGER_PARAMS);


	for (int i = 0; i < NUM_HAND_FINGER_PARAMS; ++i)
	{
		int index = NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS + i;

		float Params_Max = mHandModel->Hands_Pose_Max[i];
		float Params_Min = mHandModel->Hands_Pose_Min[i];

		if (this->Hand_Params[index] > Params_Max) {
			e_limit(i) = (Params_Max - this->Hand_Params[index]) - std::numeric_limits<float>::epsilon();
			J_limit(i, i) = 1;
		}
		else if (this->Hand_Params[index] < Params_Min) {
			e_limit(i) = (Params_Min - this->Hand_Params[index]) + std::numeric_limits<float>::epsilon();
			J_limit(i, i) = 1;
		}
		else {
			e_limit(i) = 0;
			J_limit(i, i) = 0;
		}
	}

	Eigen::MatrixXf JtJ = J_limit.transpose()*J_limit;
	Eigen::VectorXf JTe = J_limit.transpose()*e_limit;

	linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS, 
		NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
		NUM_HAND_FINGER_PARAMS,NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_MaxMinLimit_weight*JtJ;
	(linear_system.rhs.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS)).tail(NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_MaxMinLimit_weight*JTe;
}

void Worker::PcaLimit(LinearSystem& linear_system)
{
	Eigen::MatrixXf JtJ = Eigen::MatrixXf::Zero(NUM_HAND_FINGER_PARAMS, NUM_HAND_FINGER_PARAMS);
	Eigen::VectorXf JTe = Eigen::VectorXf::Zero(NUM_HAND_FINGER_PARAMS);


	//{
	//	//这部分应该是随着kalman的整合变化的，无论是 均值 还是 方差
	//	Eigen::MatrixXf shape_Sigma = Eigen::MatrixXf::Identity(this->Shape_params_num, this->Shape_params_num);
	//	shape_Sigma.diagonal() = handmodel->Hand_Shape_var;
	//	Eigen::MatrixXf InvShape_Sigma = shape_Sigma.inverse();

	//	Eigen::MatrixXf J_shape_PCA = InvShape_Sigma;
	//	Eigen::MatrixXf e_shape_PCA = -1 * InvShape_Sigma * this->Params.head(this->Shape_params_num);

	//	JtJ.block(0, 0, this->Shape_params_num, this->Shape_params_num) = settings->Shape_PcaLimit_weight * J_shape_PCA.transpose() * J_shape_PCA;
	//	JTe.head(this->Shape_params_num) = settings->Shape_PcaLimit_weight * J_shape_PCA.transpose() * e_shape_PCA;
	//}


	{
		Eigen::VectorXf Params_fingers = this->Hand_Params.tail(NUM_HAND_FINGER_PARAMS);
		Eigen::VectorXf Params_fingers_Minus_Mean = Params_fingers - mHandModel->Hands_mean;

		Eigen::MatrixXf P = mHandModel->Hands_components.transpose();


		Eigen::MatrixXf pose_Sigma = Eigen::MatrixXf::Identity(NUM_HAND_FINGER_PARAMS,NUM_HAND_FINGER_PARAMS);
		pose_Sigma.diagonal() = mHandModel->Hand_Pose_var;
		Eigen::MatrixXf Invpose_Sigma = pose_Sigma.inverse();


		Eigen::MatrixXf J_Pose_Pca = Invpose_Sigma * P;
		Eigen::VectorXf e_Pose_Pca = -1 * Invpose_Sigma*P*Params_fingers_Minus_Mean;

		JtJ = setting->Hand_Pose_Pcalimit_weight*J_Pose_Pca.transpose()*J_Pose_Pca;
		JTe = setting->Hand_Pose_Pcalimit_weight*J_Pose_Pca.transpose()*e_Pose_Pca;
	}

	linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
		NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
		NUM_HAND_FINGER_PARAMS, NUM_HAND_FINGER_PARAMS) += JtJ;
	(linear_system.rhs.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS)).tail(NUM_HAND_FINGER_PARAMS) += JTe;
}

void Worker::GloveDifferenceMaxMinPCALimit(LinearSystem& linear_system)
{
	int K = mHandModel->K_PCAcomponnet;

	Eigen::MatrixXf J_GloveDifferenceMaxMinPCALimit = Eigen::MatrixXf::Zero(K, NUM_HAND_FINGER_PARAMS);
	Eigen::VectorXf e_GloveDifferenceMaxMinPCALimit = Eigen::VectorXf::Zero(K);

	Eigen::VectorXf Params_fingers = this->Hand_Params.tail(NUM_HAND_FINGER_PARAMS);
	Eigen::VectorXf Params_glove_fingers = this->Glove_params.tail(NUM_HAND_FINGER_PARAMS);

	//下面进行PCA变换
	Eigen::VectorXf Params_fingers_Pca = mHandModel->Hands_components.leftCols(K).transpose() * (Params_fingers - mHandModel->Hands_mean);
	Eigen::VectorXf Params_glove_fingers_Pca = mHandModel->Hands_components.leftCols(K).transpose() * (Params_glove_fingers - mHandModel->Hands_mean);

	Eigen::VectorXf Params_finger_PCA_Difference = Params_fingers_Pca - Params_glove_fingers_Pca;

	for (int i = 0; i < K; ++i)
	{
		//分别计算每一个维度
		float params_difference_PCA_MAX = mHandModel->Glove_Difference_MaxPCA(i);
		float params_difference_PCA_Min = mHandModel->Glove_Difference_MinPCA(i);

		if (Params_finger_PCA_Difference(i) > params_difference_PCA_MAX) {
			e_GloveDifferenceMaxMinPCALimit(i) = (params_difference_PCA_MAX - Params_finger_PCA_Difference(i)) - std::numeric_limits<float>::epsilon();
			J_GloveDifferenceMaxMinPCALimit.row(i) = mHandModel->Hands_components.col(i).transpose();
		}
		else if (Params_finger_PCA_Difference(i) < params_difference_PCA_Min) {
			e_GloveDifferenceMaxMinPCALimit(i) = (params_difference_PCA_Min - Params_finger_PCA_Difference(i)) + std::numeric_limits<float>::epsilon();
			J_GloveDifferenceMaxMinPCALimit.row(i) = mHandModel->Hands_components.col(i).transpose();
		}
	}

	linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
		NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS, 
		NUM_HAND_FINGER_PARAMS,NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_Difference_MAXMIN_PCA_weight*J_GloveDifferenceMaxMinPCALimit.transpose() * J_GloveDifferenceMaxMinPCALimit;
	(linear_system.rhs.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS)).tail(NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_Difference_MAXMIN_PCA_weight*J_GloveDifferenceMaxMinPCALimit.transpose() * e_GloveDifferenceMaxMinPCALimit;
}

void Worker::GloveDifferenceVarPCALimit(LinearSystem& linear_system)
{
	int K = mHandModel->K_PCAcomponnet;

	Eigen::VectorXf Params_fingers = this->Hand_Params.tail(NUM_HAND_FINGER_PARAMS);
	Eigen::VectorXf Params_glove_fingers = this->Glove_params.tail(NUM_HAND_FINGER_PARAMS);

	//下面进行PCA变换
	Eigen::VectorXf Params_fingers_Pca = mHandModel->Hands_components.leftCols(K).transpose() * (Params_fingers - mHandModel->Hands_mean);
	Eigen::VectorXf Params_glove_fingers_Pca = mHandModel->Hands_components.leftCols(K).transpose() * (Params_glove_fingers - mHandModel->Hands_mean);

	Eigen::VectorXf Params_finger_PCA_Difference = Params_fingers_Pca - Params_glove_fingers_Pca;


	//下面计算雅各比
	Eigen::MatrixXf P = mHandModel->Hands_components.leftCols(K).transpose();

	Eigen::MatrixXf pose_Sigma = Eigen::MatrixXf::Identity(K, K);
	pose_Sigma.diagonal() = mHandModel->Glove_Difference_VARPCA;
	Eigen::MatrixXf Invpose_Sigma = pose_Sigma.inverse();

	Eigen::MatrixXf J_GloveDifferenceVarPCALimit = Invpose_Sigma * P;
	Eigen::VectorXf e_GloveDifferenceVarPCALimit = -1 * Invpose_Sigma*Params_finger_PCA_Difference;

	linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
		NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
		NUM_HAND_FINGER_PARAMS, NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_Difference_Var_PCA_weight*J_GloveDifferenceVarPCALimit.transpose() * J_GloveDifferenceVarPCALimit;
	(linear_system.rhs.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS)).tail(NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_Difference_Var_PCA_weight*J_GloveDifferenceVarPCALimit.transpose() * e_GloveDifferenceVarPCALimit;
}

void Worker::TemporalLimit(LinearSystem& linear_system, bool first_order)
{
	//先弄人手
	{
		Eigen::MatrixXf J_Tem = Eigen::MatrixXf::Identity(NUM_HAND_POSE_PARAMS, NUM_HAND_POSE_PARAMS);
		Eigen::VectorXf e_Tem = Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS);

		if (temporal_finger_params.size() == 2)
		{
			for (int i = 0; i < NUM_HAND_POSE_PARAMS; ++i)
			{
				int index = NUM_HAND_SHAPE_PARAMS + i;

				if (first_order)
				{
					e_Tem(i) = temporal_finger_params.back()(i) - Hand_Params(index);
				}
				else
				{
					e_Tem(i) = 2 * temporal_finger_params.back()(i) - temporal_finger_params.front()(i) - Hand_Params(index);
				}
			}
		}

		if (first_order)
		{
			linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS,NUM_HAND_SHAPE_PARAMS,
				NUM_HAND_POSE_PARAMS,NUM_HAND_POSE_PARAMS) += setting->Temporal_finger_params_FirstOrder_weight*J_Tem.transpose()*J_Tem;
			(linear_system.rhs.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS)).tail(NUM_HAND_POSE_PARAMS) += setting->Temporal_finger_params_FirstOrder_weight*J_Tem.transpose()*e_Tem;
		}
		else
		{
			linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS,
				NUM_HAND_POSE_PARAMS, NUM_HAND_POSE_PARAMS) += setting->Temporal_finger_params_SecondOorder_weight*J_Tem.transpose()*J_Tem;
			(linear_system.rhs.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS)).tail(NUM_HAND_POSE_PARAMS) += setting->Temporal_finger_params_SecondOorder_weight*J_Tem.transpose()*e_Tem;
		}
	}

	//再弄物体
	{
		Eigen::MatrixXf J_Tem = Eigen::MatrixXf::Identity(NUM_OBJECT_PARAMS, NUM_OBJECT_PARAMS);
		Eigen::VectorXf e_Tem = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);

		Eigen::MatrixXf joint_jacob;

		if (temporal_Object_params.size() == 2)
		{
			for (int i = 0; i < NUM_OBJECT_PARAMS; ++i)
			{
				if (first_order)
				{
					e_Tem(i) = temporal_Object_params.back()(i) - Object_params(i);
				}
				else
				{
					e_Tem(i) = 2 * temporal_Object_params.back()(i) - temporal_Object_params.front()(i) - Object_params(i);
				}
			}
		}

		if (first_order)
		{
			linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
				NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
				NUM_OBJECT_PARAMS,NUM_OBJECT_PARAMS) += setting->Object_Temporal_firstOrder_weight*J_Tem.transpose()*J_Tem;
			linear_system.rhs.tail(NUM_OBJECT_PARAMS) += setting->Object_Temporal_firstOrder_weight*J_Tem.transpose()*e_Tem;
		}
		else
		{
			linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
				NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
				NUM_OBJECT_PARAMS, NUM_OBJECT_PARAMS) += setting->Object_Temporal_secondOrder_weight*J_Tem.transpose()*J_Tem;
			linear_system.rhs.tail(NUM_OBJECT_PARAMS) += setting->Object_Temporal_secondOrder_weight*J_Tem.transpose()*e_Tem;
		}
	}
}

void Worker::CollisionLimit(LinearSystem& linear_system)
{
	int NumOfCollision = mHandModel->NumOfCollision;
	int CollisionSphereNum = mHandModel->Collision_sphere.size();

	float fraction = 1.0f;
	Eigen::MatrixXf jacob_tmp;

	if (NumOfCollision > 0)
	{
		Eigen::MatrixXf J_collision = Eigen::MatrixXf::Zero(3 * NumOfCollision, NUM_HAND_POSE_PARAMS);
		Eigen::VectorXf e_collision = Eigen::VectorXf::Zero(3 * NumOfCollision);

		int count = 0;

		for (int i = 0; i < CollisionSphereNum; ++i)
		{
			for (int j = 0; j < CollisionSphereNum; ++j)
			{
				if (mHandModel->Collision_Judge_Matrix(i, j) == 1) //发生碰撞，规则为 i 和 j 碰撞，则i 需要移动（后面有 j 和 i 碰撞，j需要移动）
				{
					Eigen::Vector3f dir_i_to_j;

					dir_i_to_j << mHandModel->Collision_sphere[j].Update_Center - mHandModel->Collision_sphere[i].Update_Center;
					dir_i_to_j.normalize();

					Eigen::Vector3f now_point, target_point;

					now_point = mHandModel->Collision_sphere[i].Update_Center + mHandModel->Collision_sphere[i].Radius*dir_i_to_j;
					target_point = mHandModel->Collision_sphere[j].Update_Center - mHandModel->Collision_sphere[j].Radius*dir_i_to_j;

					e_collision(count * 3 + 0) = fraction * (target_point(0) - now_point(0));
					e_collision(count * 3 + 1) = fraction * (target_point(1) - now_point(1));
					e_collision(count * 3 + 2) = fraction * (target_point(2) - now_point(2));

					Eigen::Vector3f now_point_Initpos = now_point - mHandModel->Collision_sphere[i].Update_Center + mHandModel->Collision_sphere[i].Init_Center;

					mHandModel->CollisionPoint_Jacobian(jacob_tmp, mHandModel->Collision_sphere[i].joint_belong, now_point_Initpos);

					J_collision.block(count * 3, 0, 3, NUM_HAND_POSE_PARAMS) = jacob_tmp;
					count++;
				}
			}
		}

		linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS, 
			NUM_HAND_POSE_PARAMS, NUM_HAND_POSE_PARAMS) += setting->Collision_weight * J_collision.transpose() * J_collision;
		(linear_system.rhs.head(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS)).tail(NUM_HAND_POSE_PARAMS) += setting->Collision_weight * J_collision.transpose() * e_collision;
	}
}

void Worker::RigidOnly(LinearSystem& linear_system)
{
	for (int row = 0; row < NUM_HAND_SHAPE_PARAMS; ++row)
	{
		linear_system.lhs.row(row).setZero();
		linear_system.rhs.row(row).setZero();
	}
	for (int row = NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS; row < NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS; ++row)
	{
		linear_system.lhs.row(row).setZero();
		linear_system.rhs.row(row).setZero();
	}


	for (int col = 0; col < NUM_HAND_SHAPE_PARAMS; ++col) linear_system.lhs.col(col).setZero();
	for (int col = NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS; col < NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS; ++col) linear_system.lhs.col(col).setZero();

}

void Worker::Damping(LinearSystem& linear_system)
{
	//先弄人手的
	{
		Eigen::MatrixXf D = Eigen::MatrixXf::Identity(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS, NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);

		D.block(0, 0, NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS) = setting->Hand_Shape_Damping_weight * Eigen::MatrixXf::Identity(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS);

		for (int i = 0; i < 48; ++i)
		{
			if (i == 3 || i == 6 || i == 7 || i == 9 || i == 10 ||
				i == 12 || i == 15 || i == 16 || i == 18 || i == 19 ||
				i == 21 || i == 24 || i == 25 || i == 27 || i == 28 ||
				i == 30 || i == 33 || i == 34 || i == 36 || i == 37 ||
				i == 39 || i == 42 || i == 44 || i == 45 || i == 47)
			{
				int index = NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSITION_PARAMS + i;

				D(index, index) = setting->Hand_Pose_Damping_weight_For_BigRotate;
			}

			if (i == 4 || i == 5 || i == 8 || i == 11 ||
				i == 13 || i == 14 || i == 17 || i == 20 ||
				i == 22 || i == 23 || i == 26 || i == 29 ||
				i == 31 || i == 32 || i == 35 || i == 38 ||
				i == 40 || i == 41 || i == 43 || i == 46)
			{
				int index = NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSITION_PARAMS + i;

				D(index, index) = setting->Hand_Pose_Damping_weight_For_SmallRotate;
			}
		}
		linear_system.lhs.block(0,0,
			NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
			NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS) += D;
	}

	//再弄物体的
	{
		Eigen::MatrixXf D = Eigen::MatrixXf::Identity(NUM_OBJECT_PARAMS, NUM_OBJECT_PARAMS);

		D(0, 0) = setting->Object_Trans_Damping;
		D(1, 1) = setting->Object_Trans_Damping;
		D(2, 2) = setting->Object_Trans_Damping;

		D(3, 3) = setting->Object_Rotate_Damping;
		D(4, 4) = setting->Object_Rotate_Damping;
		D(5, 5) = setting->Object_Rotate_Damping;

linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
	NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
	NUM_OBJECT_PARAMS, NUM_OBJECT_PARAMS) += D;
	}
}

Eigen::VectorXf Worker::Solver(LinearSystem& linear_system)
{
	///--- Solve for update dt = (J^T * J + D)^-1 * J^T * r

	//http://eigen.tuxfamily.org/dox/group__LeastSquares.html  for linear least square problem
	Eigen::VectorXf solution = linear_system.lhs.colPivHouseholderQr().solve(linear_system.rhs);

	///--- Check for NaN
	for (int i = 0; i < solution.size(); i++) {
		if (isnan(solution[i])) {
			std::cout << "-------------------------------------------------------------\n";
			std::cout << "!!!WARNING: NaN DETECTED in the solution!!! (skipping update)\n";
			std::cout << "-------------------------------------------------------------\n";
			return Eigen::VectorXf::Zero(solution.size());
		}

		if (isinf(solution[i])) {
			std::cout << "-------------------------------------------------------------\n";
			std::cout << "!!!WARNING: INF DETECTED in the solution!!! (skipping update)\n";
			std::cout << "-------------------------------------------------------------\n";
			return Eigen::VectorXf::Zero(solution.size());
		}
	}

	return solution;
}

void Worker::Evaluation()
{
	float errors = 0.0f;
	int count = 0;
	float MaxThreshold = 100.0f;

	int rows = mCamera->height();
	int cols = mCamera->width();

	//先通过人手和物体的生成的深度图合成一个总体的深度图，根据深度前后关系进行合并
	mInteracted_Object->GenerateDepthAndSilhouette();
	mHandModel->GenerateDepthMap();

	mRendered_Images.setToZero();

	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			if (mInteracted_Object->generatedSilhouette.at<uchar>(row, col) == 255
				&& mHandModel->HandModel_binaryMap.at<uchar>(row, col) != 255)
			{
				mRendered_Images.total_silhouette.at<uchar>(row, col) = 255;
				mRendered_Images.total_depth.at<ushort>(row, col) = mInteracted_Object->generatedDepth.at<ushort>(row, col);

				count++;
				float tmp_error = abs(mRendered_Images.total_depth.at<ushort>(row, col) - mImage_InputData->depth.at<ushort>(row, col));
				errors += tmp_error > MaxThreshold ? MaxThreshold : tmp_error;
			}

			if (mInteracted_Object->generatedSilhouette.at<uchar>(row, col) != 255
				&& mHandModel->HandModel_binaryMap.at<uchar>(row, col) == 255)
			{
				mRendered_Images.total_silhouette.at<uchar>(row, col) = 255;
				mRendered_Images.total_depth.at<ushort>(row, col) = mHandModel->HandModel_depthMap.at<ushort>(row, col);

				count++;
				float tmp_error = abs(mRendered_Images.total_depth.at<ushort>(row, col) - mImage_InputData->depth.at<ushort>(row, col));
				errors += tmp_error > MaxThreshold ? MaxThreshold : tmp_error;
			}

			if (mInteracted_Object->generatedSilhouette.at<uchar>(row, col) == 255
				&& mHandModel->HandModel_binaryMap.at<uchar>(row, col) == 255)
			{
				mRendered_Images.total_silhouette.at<uchar>(row, col) = 255;
				ushort Object_depth = mInteracted_Object->generatedDepth.at<ushort>(row, col);
				ushort Hand_depth = mHandModel->HandModel_depthMap.at<ushort>(row, col);

				if (Object_depth > Hand_depth)
				{
					mRendered_Images.total_depth.at<ushort>(row, col) = Hand_depth;
					mInteracted_Object->generatedSilhouette.at<uchar>(row, col) = 0;
				}
				else
				{
					mRendered_Images.total_depth.at<ushort>(row, col) = Object_depth;
					mHandModel->HandModel_binaryMap.at<uchar>(row, col) = 0;
				}

				count++;
				float tmp_error = abs(mRendered_Images.total_depth.at<ushort>(row, col) - mImage_InputData->depth.at<ushort>(row, col));
				errors += tmp_error > MaxThreshold ? MaxThreshold : tmp_error;
			}

			if (mInteracted_Object->generatedSilhouette.at<uchar>(row, col) != 255 &&
				mHandModel->HandModel_binaryMap.at<uchar>(row, col) != 255 &&
				mImage_InputData->silhouette.at<uchar>(row, col) == 255)
			{
				count++;
				float tmp_error = abs(0 - mImage_InputData->depth.at<ushort>(row, col));
				errors += tmp_error > MaxThreshold ? MaxThreshold : tmp_error;
			}
		}
	}

	mInteracted_Object->generatedDepth.copyTo(mRendered_Images.rendered_object_depth);
	mInteracted_Object->generatedSilhouette.copyTo(mRendered_Images.rendered_object_silhouette);
	mHandModel->HandModel_binaryMap.copyTo(mRendered_Images.rendered_hand_silhouette);
	mHandModel->HandModel_depthMap.copyTo(mRendered_Images.rendered_hand_depth);

	if (count <= 0)
		tracking_success = false;
	else
	{
		tracking_error = errors / count;
		tracking_success = tracking_error > setting->tracking_fail_threshold ? false : true;
	}
}