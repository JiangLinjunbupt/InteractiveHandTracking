#include"Worker.h"

Worker::Worker(Camera* _camera, vector<Object_type>& object_type)
{
	mCamera = _camera;

	//��ʼ���������
	mHandModel = new HandModel(mCamera);
	Hand_Params = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);
	mHandModel->set_Shape_Params(Hand_Params.head(NUM_HAND_SHAPE_PARAMS));
	mHandModel->set_Pose_Params(Hand_Params.tail(NUM_HAND_POSE_PARAMS));
	mHandModel->UpdataModel();

	//��ʼ���������
	for (size_t obj_id = 0; obj_id < object_type.size(); ++obj_id)
	{
		Interacted_Object* tmpObject = nullptr;
		switch (object_type[obj_id])
		{
		case yellowSphere:
			tmpObject = new YellowSphere(mCamera);
			break;
		case redCube:
			tmpObject = new RedCube(mCamera);
			break;
		default:
			break;
		}
		tmpObject->Update(Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS));
		mInteracted_Objects.push_back(tmpObject);
		Object_params.emplace_back(Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS));
	}


	//��ʼ��������ز���
	kalman = new Kalman(mHandModel);
	mRendered_Images.init(mCamera->width(), mCamera->height());

	tracking_error = 0.0f;
	tracking_success = false;
	itr = 0;
	total_itr = 0;
	mImage_InputData = nullptr;
	Glove_params = Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS);

	Hand_correspond.clear();
	Object_corresponds.resize(object_type.size());
	num_Object_matched_correspond.resize(object_type.size());
	temporal_Object_params.resize(object_type.size());
	//------------------��Ҫ��������
	for(int i = 0;i<temporal_Object_params.size();++i)
		while (!temporal_Object_params[i].empty()) temporal_Object_params[i].pop();
	while (!temporal_Hand_params.empty()) temporal_Hand_params.pop();
}

void Worker::Tracking(Image_InputData& imageData, Glove_InputData& gloveData,
	Eigen::VectorXf& hand_init, vector<Eigen::VectorXf>& object_init,
	bool pre_success,
	Eigen::VectorXf& pre_handPrams, vector<Eigen::VectorXf>& pre_objectParams)
{
	if (pre_success)
		setting->max_itr = 6;
	else
		setting->max_itr = 15;

	tracking_success = false;
	itr = 0;

	Has_Glove = true;
	//�����������ݣ�������ģ����ʼ��
	mImage_InputData = &imageData;
	Glove_params = gloveData.params;

	SetHandInit(hand_init, pre_success, pre_handPrams);
	SetObjectsInit(object_init, pre_success, pre_objectParams);
	SetTemporalInfo(pre_success, pre_handPrams, pre_objectParams);
	for (; itr < setting->max_itr; ++itr)
	{
		FindObjectCorrespond();
		FindHandCorrespond();

		Hand_one_tracking();
		for (int obj_idx = 0; obj_idx < mInteracted_Objects.size(); ++obj_idx)
		{
			if(!mImage_InputData->item[obj_idx].loss_detect)
				Object_one_tracking(obj_idx);
		}

		total_itr++;
	}

	Evaluation();
}
void Worker::SetHandInit(Eigen::VectorXf& hand_init, bool pre_success, Eigen::VectorXf& pre_handParams)
{
	if (pre_success) Hand_Params.head(NUM_HAND_SHAPE_PARAMS) = pre_handParams.head(NUM_HAND_SHAPE_PARAMS);
	else
	{
		Hand_Params.head(NUM_HAND_SHAPE_PARAMS) = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
		kalman->ReSet();
	}
	Hand_Params.tail(NUM_HAND_POSE_PARAMS) = hand_init;
	mHandModel->set_Pose_Params(Hand_Params.tail(NUM_HAND_POSE_PARAMS));
	mHandModel->set_Shape_Params(Hand_Params.head(NUM_HAND_SHAPE_PARAMS));
	mHandModel->UpdataModel();
}
void Worker::SetObjectsInit(vector<Eigen::VectorXf>& object_init, bool pre_success, vector<Eigen::VectorXf>& pre_objectParams)
{
	//���֮ǰ���ٳɹ������Ҳ��ǵ�һ�μ�⵽�����壬��ʹ��ǰһ֡�ʵĲ���������ʹ�ó�ʼ����֡��
	for (size_t obj_id = 0; obj_id < mInteracted_Objects.size(); ++obj_id)
	{
		if (pre_success && (!mImage_InputData->item[obj_id].first_detect) && mImage_InputData->item[obj_id].now_detect)
		{
			Object_params[obj_id] = pre_objectParams[obj_id];
			mInteracted_Objects[obj_id]->Update(Object_params[obj_id]);
		}
		else
		{
			Object_params[obj_id] = object_init[obj_id];
			mInteracted_Objects[obj_id]->Update(Object_params[obj_id]);
		}
	}
}

void Worker::SetTemporalInfo(bool pre_success, Eigen::VectorXf& pre_handPrams, vector<Eigen::VectorXf>& pre_objectParams)
{
	if (pre_success)
	{
		for (int obj_idx = 0; obj_idx < mInteracted_Objects.size(); ++obj_idx)
		{
			if (temporal_Object_params[obj_idx].size() == 2) {
				temporal_Object_params[obj_idx].pop();
				temporal_Object_params[obj_idx].push(pre_objectParams[obj_idx]);
			}
			else
			{
				temporal_Object_params[obj_idx].push(pre_objectParams[obj_idx]);
			}
		}

		if (temporal_Hand_params.size() == 2)
		{
			temporal_Hand_params.pop();
			temporal_Hand_params.push(pre_handPrams.tail(NUM_HAND_FINGER_PARAMS));
		}
		else
		{
			temporal_Hand_params.push(pre_handPrams.tail(NUM_HAND_FINGER_PARAMS));
		}
	}
	else
	{
		for (int obj_idx = 0; obj_idx<temporal_Object_params.size(); ++obj_idx)
			while (!temporal_Object_params[obj_idx].empty()) temporal_Object_params[obj_idx].pop();
		while (!temporal_Hand_params.empty())
			temporal_Hand_params.pop();
	}
}

void Worker::FindObjectCorrespond()
{
	for (int obj_id = 0; obj_id < mInteracted_Objects.size(); ++obj_id)
	{
		Object_corresponds[obj_id].clear();

		pcl::PointCloud<pcl::PointNormal> object_visible_cloud;
		std::vector<int> visible_idx;
		int Maxsize = mInteracted_Objects[obj_id]->Final_Vertices.size();

		object_visible_cloud.reserve(Maxsize);
		visible_idx.reserve(Maxsize);

		for (int i = 0; i < Maxsize; ++i)
		{
			if (mInteracted_Objects[obj_id]->Final_Normal[i].z() < 0)
			{
				pcl::PointNormal p;
				p.x = mInteracted_Objects[obj_id]->Final_Vertices[i](0);
				p.y = mInteracted_Objects[obj_id]->Final_Vertices[i](1);
				p.z = mInteracted_Objects[obj_id]->Final_Vertices[i](2);

				p.normal_x = mInteracted_Objects[obj_id]->Final_Normal[i](0);
				p.normal_y = mInteracted_Objects[obj_id]->Final_Normal[i](1);
				p.normal_z = mInteracted_Objects[obj_id]->Final_Normal[i](2);

				object_visible_cloud.points.push_back(p);
				visible_idx.push_back(i);
			}
		}

		//Ȼ�����Ҷ�Ӧ��
		int Numvisible = object_visible_cloud.size();
		int NumPointCloud_sensor = mImage_InputData->item[obj_id].pointcloud.points.size();

		if (Numvisible > 0 && NumPointCloud_sensor > 0)
		{
			Object_corresponds[obj_id].resize(NumPointCloud_sensor);
			pcl::KdTreeFLANN<pcl::PointNormal> search_tree;
			search_tree.setInputCloud(object_visible_cloud.makeShared());

			const int k = 1;
			std::vector<int> k_indices(k);
			std::vector<float> k_squared_distance(k);

			for (int i = 0; i < NumPointCloud_sensor; ++i)
			{
				search_tree.nearestKSearch(mImage_InputData->item[obj_id].pointcloud, i, k, k_indices, k_squared_distance);

				Eigen::Vector3f p = Eigen::Vector3f(mImage_InputData->item[obj_id].pointcloud.points[i].x,
					mImage_InputData->item[obj_id].pointcloud.points[i].y,
					mImage_InputData->item[obj_id].pointcloud.points[i].z);
				Object_corresponds[obj_id][i].pointcloud = p;


				Eigen::Vector3f p_cor = Eigen::Vector3f(object_visible_cloud.points[k_indices[0]].x,
					object_visible_cloud.points[k_indices[0]].y,
					object_visible_cloud.points[k_indices[0]].z);

				Object_corresponds[obj_id][i].correspond = p_cor;
				Object_corresponds[obj_id][i].correspond_idx = visible_idx[k_indices[0]];

				float distance = (p_cor - p).norm();

				if (distance > 100)
					Object_corresponds[obj_id][i].is_match = false;
				else
					Object_corresponds[obj_id][i].is_match = true;
			}
		}

		num_Object_matched_correspond[obj_id] = 0;

		for (std::vector<DataAndCorrespond>::iterator itr = Object_corresponds[obj_id].begin(); itr != Object_corresponds[obj_id].end(); ++itr)
		{
			if (itr->is_match)
				++num_Object_matched_correspond[obj_id];
		}
	}
}

void Worker::FindHandCorrespond()
{
	Hand_correspond.clear();
	//���ȼ���handmodel�е�visible point
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


	//Ȼ�����Ҷ�Ӧ��
	int NumVisible_ = Handmodel_visible_cloud.points.size();
	int NumPointCloud_sensor = mImage_InputData->hand.pointcloud.points.size();

	if (NumVisible_ > 0 && NumPointCloud_sensor > 0)
	{
		Hand_correspond.resize(NumPointCloud_sensor);

		pcl::KdTreeFLANN<pcl::PointNormal> search_kdtree;
		search_kdtree.setInputCloud(Handmodel_visible_cloud.makeShared());  //����ע��PCL��flann���opencv��flann��ͻ��ע����

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
			Hand_correspond[i].pointcloud_n = Eigen::Vector3f(mImage_InputData->hand.pointcloud.points[i].normal_x,
				mImage_InputData->hand.pointcloud.points[i].normal_y,
				mImage_InputData->hand.pointcloud.points[i].normal_z);


			Eigen::Vector3f p_2 = Eigen::Vector3f(Handmodel_visible_cloud.points[k_indices[0]].x,
				Handmodel_visible_cloud.points[k_indices[0]].y,
				Handmodel_visible_cloud.points[k_indices[0]].z);

			Hand_correspond[i].correspond = p_2;
			Hand_correspond[i].correspond_n = Eigen::Vector3f(Handmodel_visible_cloud.points[k_indices[0]].normal_x,
				Handmodel_visible_cloud.points[k_indices[0]].normal_y,
				Handmodel_visible_cloud.points[k_indices[0]].normal_z);
			Hand_correspond[i].correspond_idx = visible_idx[k_indices[0]];

			float distance = (p_2 - p).norm();

			if (distance > 50)
				Hand_correspond[i].is_match = false;
			else
				Hand_correspond[i].is_match = true;
		}
	}

	int NumHandVisible_ = mHandModel->Visible_3D.size();
	if (NumHandVisible_ > 0 && NumPointCloud_sensor > 0 && itr == (setting->max_itr - 1))
	{
		pcl::KdTreeFLANN<pcl::PointNormal> search_kdtree;
		search_kdtree.setInputCloud(mImage_InputData->hand.pointcloud.makeShared());  //����ע��PCL��flann���opencv��flann��ͻ��ע����

		const int k = 1;
		std::vector<int> k_indices(k);
		std::vector<float> k_squared_distances(k);
		for (int i = 0; i < NumHandVisible_; ++i)
		{
			search_kdtree.nearestKSearch(mHandModel->Visible_3D, i, k, k_indices, k_squared_distances);

			DataAndCorrespond tmp;
			Eigen::Vector3f p = Eigen::Vector3f(mImage_InputData->hand.pointcloud.points[k_indices[0]].x,
				mImage_InputData->hand.pointcloud.points[k_indices[0]].y,
				mImage_InputData->hand.pointcloud.points[k_indices[0]].z);
			tmp.pointcloud = p;
			tmp.pointcloud_n = Eigen::Vector3f(mImage_InputData->hand.pointcloud.points[k_indices[0]].normal_x,
				mImage_InputData->hand.pointcloud.points[k_indices[0]].normal_y,
				mImage_InputData->hand.pointcloud.points[k_indices[0]].normal_z);

			Eigen::Vector3f p_2 = Eigen::Vector3f(mHandModel->Visible_3D.points[i].x,
				mHandModel->Visible_3D.points[i].y,
				mHandModel->Visible_3D.points[i].z);

			tmp.correspond = p_2;
			tmp.correspond_n = Eigen::Vector3f(mHandModel->Visible_3D.points[i].normal_x,
				mHandModel->Visible_3D.points[i].normal_y,
				mHandModel->Visible_3D.points[i].normal_z);
			tmp.correspond_idx = mHandModel->V_Visible_2D[i].second;

			float distance = (p_2 - p).norm();

			if (distance > 50)
				tmp.is_match = false;
			else
				tmp.is_match = true;

			Hand_correspond.emplace_back(tmp);

		}
	}

	num_Hand_matched_correspond = 0;
	for(vector<DataAndCorrespond>::iterator itr = Hand_correspond.begin();itr!=Hand_correspond.end();++itr)
	{
		if (itr->is_match)
			++num_Hand_matched_correspond;
	}
}

void Worker::Hand_one_tracking()
{
	LinearSystem linear_system;
	linear_system.lhs = Eigen::MatrixXf::Zero(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS,
		NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);
	linear_system.rhs = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS);

	float error_3D = this->Fitting3D(linear_system);
	this->Fitting2D(linear_system);

	//������Կ�ʼ����kalman�ĺ�ɭ����
	if (itr>(setting->max_itr - 2)
		&& error_3D < 8.0f
		&& !kalman->judgeFitted())  //�����ڸ��µĵ����󼸴Σ����Ҹ��ٳɹ���ʱ�����
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
	Hand_Object_Collision(linear_system);
	this->Damping(linear_system);
	if (itr < setting->max_rigid_itr)
		this->RigidOnly(linear_system);  //���һ��Ҫ�����

	//���
	Eigen::VectorXf solution = this->Solver(linear_system);

	for (int i = 0; i < (NUM_HAND_SHAPE_PARAMS + NUM_HAND_POSE_PARAMS); ++i)
		Hand_Params[i] += solution[i];

	//�������ͨ��total_itr���������жϣ�ͨ��kalman������״�����ļ��
	if (total_itr > 2 * setting->frames_interval_between_measurements
		&& (total_itr % setting->frames_interval_between_measurements) == 0
		&& error_3D < 8.0f
		&& !kalman->judgeFitted())
		kalman->Update_estimate(Hand_Params.head(NUM_HAND_SHAPE_PARAMS));

	//���²���
	mHandModel->set_Shape_Params(Hand_Params.head(NUM_HAND_SHAPE_PARAMS));
	mHandModel->set_Pose_Params(Hand_Params.tail(NUM_HAND_POSE_PARAMS));
	mHandModel->UpdataModel();
}

void Worker::Object_one_tracking(int obj_idx)
{
	//��ʼ��������
	LinearSystem linear_system;
	linear_system.lhs = Eigen::MatrixXf::Zero(NUM_OBJECT_PARAMS,NUM_OBJECT_PARAMS);
	linear_system.rhs = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);


	this->Object_Fitting_3D(linear_system, obj_idx);
	this->Object_Fitting_2D(linear_system, obj_idx);
	this->Object_TemporalLimit(linear_system, true, obj_idx);
	this->Object_TemporalLimit(linear_system, false, obj_idx);
	this->Object_CollisionLimit(linear_system, obj_idx);
	this->Object_Damping(linear_system);

	//���
	Eigen::VectorXf solution = this->Solver(linear_system);

	for (int i = 0; i < NUM_OBJECT_PARAMS; ++i)
		Object_params[obj_idx][i] += solution[i];

	mInteracted_Objects[obj_idx]->Update(Object_params[obj_idx]);
}

float Worker::Fitting3D(LinearSystem& linear_system)
{
	float hand_3D_error = 0.0f;

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
			//����ʹ�õ���Reweighted Least squard error
			//�ο���
			//https://www.cs.bgu.ac.il/~mcv172/wiki.files/Lec5.pdf ����Ҫ��
			//https://blog.csdn.net/baidu_17640849/article/details/71155537  ��������
			//weight = s/(e^2 + s^2)


			//sԽ�󣬶��쳣ֵ������Խ��
			float s = 50;
			float weight = 1;
			weight = (itr + 1) * s / (e_sqrt + s);

			weight = 6.5f / sqrt(e_sqrt + 0.001);

			weight *= mHandModel->vertices_fitting_weight[v_id];
			weight *= Hand_correspond[i].correspond_n.dot(Hand_correspond[i].pointcloud_n);

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

	linear_system.lhs += setting->Hand_fitting_3D*J.transpose()*J;
	linear_system.rhs += setting->Hand_fitting_3D*J.transpose()*e;

	return hand_3D_error;
}

void Worker::Object_Fitting_3D(LinearSystem& linear_system, int obj_idx)
{
	int NumofCorrespond = num_Object_matched_correspond[obj_idx];
	Eigen::VectorXf e = Eigen::VectorXf::Zero(NumofCorrespond * 3);
	Eigen::MatrixXf J = Eigen::MatrixXf::Zero(NumofCorrespond * 3, NUM_OBJECT_PARAMS);

	//��ʱ����
	Eigen::MatrixXf tmp_jacob;
	int count = 0;
	int Object_correspond_size = Object_corresponds[obj_idx].size();

	for (int i = 0; i < Object_correspond_size; ++i)
	{
		if (Object_corresponds[obj_idx][i].is_match)
		{
			int v_id = Object_corresponds[obj_idx][i].correspond_idx;

			e(count * 3 + 0) = Object_corresponds[obj_idx][i].pointcloud(0) - Object_corresponds[obj_idx][i].correspond(0);
			e(count * 3 + 1) = Object_corresponds[obj_idx][i].pointcloud(1) - Object_corresponds[obj_idx][i].correspond(1);
			e(count * 3 + 2) = Object_corresponds[obj_idx][i].pointcloud(2) - Object_corresponds[obj_idx][i].correspond(2);

			float e_sqrt = sqrt(pow(e(count * 3 + 0), 2) + pow(e(count * 3 + 1), 2) + pow(e(count * 3 + 2), 2));
			//����ʹ�õ���Reweighted Least squard error
			//�ο���
			//https://www.cs.bgu.ac.il/~mcv172/wiki.files/Lec5.pdf ����Ҫ��
			//https://blog.csdn.net/baidu_17640849/article/details/71155537  ��������
			//weight = s/(e^2 + s^2)


			//sԽ�󣬶��쳣ֵ������Խ��
			float s = 50;
			float weight = 1;
			weight = (itr + 1) * s / (e_sqrt + s);

			weight = 6.5f / sqrt(e_sqrt + 0.001);

			e(count * 3 + 0) *= weight;
			e(count * 3 + 1) *= weight;
			e(count * 3 + 2) *= weight;

			mInteracted_Objects[obj_idx]->object_jacobain(tmp_jacob, v_id);
			J.block(count * 3, 0, 3, NUM_OBJECT_PARAMS) = tmp_jacob;
			++count;
		}
	}

	linear_system.lhs += setting->Object_fitting_3D*J.transpose()*J;
	linear_system.rhs += setting->Object_fitting_3D*J.transpose()*e;
}

void Worker::Fitting2D(LinearSystem& linear_system)
{
	int width = mCamera->width();
	int height = mCamera->height();

	vector<pair<Eigen::Matrix2Xf, Eigen::Vector2f>> JacobVector_2D;

	int visiblePointSize = mHandModel->V_Visible_2D.size();

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
				//���� J �� e
				pair<Eigen::Matrix2Xf, Eigen::Vector2f> J_and_e;

				//����e
				Eigen::Vector2f e;
				e(0) = (float)pixel_2D_cloest(0) - (float)pixel_2D_pos(0);
				e(1) = (float)pixel_2D_cloest(1) - (float)pixel_2D_pos(1);

				J_and_e.second = e;

				//�ټ���J
				Eigen::Matrix<float, 2, 3> J_perspective = mCamera->projection_jacobian(pixel_3D_pos);
				Eigen::MatrixXf J_3D;

				mHandModel->Pose_jacobain(J_3D, idx);
				J_and_e.first = J_perspective * J_3D;

				JacobVector_2D.push_back(J_and_e);
			}
		}

	}

	int size = JacobVector_2D.size();

	if (size > 0)
	{
		Eigen::MatrixXf J_2D = Eigen::MatrixXf::Zero(2 * size, NUM_HAND_POSE_PARAMS);
		Eigen::VectorXf e_2D = Eigen::VectorXf::Zero(2 * size);

		for (int i = 0; i < size; ++i)
		{
			J_2D.block(i * 2, 0, 2, NUM_HAND_POSE_PARAMS) = JacobVector_2D[i].first;
			e_2D.segment(i * 2, 2) = JacobVector_2D[i].second;
		}

		linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS,
			NUM_HAND_POSE_PARAMS,NUM_HAND_POSE_PARAMS) += setting->Hand_fitting_2D * J_2D.transpose() * J_2D;
		linear_system.rhs.tail(NUM_HAND_POSE_PARAMS) += setting->Hand_fitting_2D * J_2D.transpose() * e_2D;
	}
}

void Worker::Object_Fitting_2D(LinearSystem& linear_system, int obj_idx)
{
	int width = mCamera->width();
	int height = mCamera->height();

	vector<pair<Eigen::Matrix2Xf, Eigen::Vector2f>> JacobVector_2D;

	int visiblePointSize = mInteracted_Objects[obj_idx]->Visible_2D.size();

	Eigen::MatrixXf tmp_jacob;

	for (int i = 0; i < visiblePointSize; ++i)
	{
		int idx = mInteracted_Objects[obj_idx]->Visible_2D[i].second;

		Eigen::Vector3f pixel_3D_pos = mInteracted_Objects[obj_idx]->Final_Vertices[idx];
		Eigen::Vector2i pixel_2D_pos = Eigen::Vector2i(mInteracted_Objects[obj_idx]->Visible_2D[i].first.x(), 
			mInteracted_Objects[obj_idx]->Visible_2D[i].first.y());

		if (pixel_2D_pos.x() >= 0 && pixel_2D_pos.x() < width && pixel_2D_pos.y() >= 0 && pixel_2D_pos.y() < height)
		{
			int cloest_idx = mImage_InputData->idxs_image[pixel_2D_pos(1) * width + pixel_2D_pos(0)];
			Eigen::Vector2i pixel_2D_cloest;
			pixel_2D_cloest << cloest_idx%width, cloest_idx / width;

			float closet_distance = (pixel_2D_cloest - pixel_2D_pos).norm();

			if (closet_distance > 0)
			{
				//���� J �� e
				pair<Eigen::Matrix2Xf, Eigen::Vector2f> J_and_e;

				//����e
				Eigen::Vector2f e;
				e(0) = (float)pixel_2D_cloest(0) - (float)pixel_2D_pos(0);
				e(1) = (float)pixel_2D_cloest(1) - (float)pixel_2D_pos(1);

				J_and_e.second = e;

				//�ټ���J
				Eigen::Matrix<float, 2, 3> J_perspective = mCamera->projection_jacobian(pixel_3D_pos);
				mInteracted_Objects[obj_idx]->object_jacobain(tmp_jacob, idx);

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

		linear_system.lhs += setting->Object_fitting_2D * J_2D.transpose() * J_2D;
		linear_system.rhs += setting->Object_fitting_2D * J_2D.transpose() * e_2D;
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

	linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS, 
		NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
		NUM_HAND_FINGER_PARAMS,NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_MaxMinLimit_weight*J_limit.transpose()*J_limit;
	linear_system.rhs.tail(NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_MaxMinLimit_weight*J_limit.transpose()*e_limit;
}

void Worker::PcaLimit(LinearSystem& linear_system)
{
	Eigen::MatrixXf JtJ = Eigen::MatrixXf::Zero(NUM_HAND_FINGER_PARAMS, NUM_HAND_FINGER_PARAMS);
	Eigen::VectorXf JTe = Eigen::VectorXf::Zero(NUM_HAND_FINGER_PARAMS);


	//{
	//	//�ⲿ��Ӧ��������kalman�����ϱ仯�ģ������� ��ֵ ���� ����
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
	linear_system.rhs.tail(NUM_HAND_FINGER_PARAMS) += JTe;
}

void Worker::GloveDifferenceMaxMinPCALimit(LinearSystem& linear_system)
{
	int K = mHandModel->K_PCAcomponnet;

	Eigen::MatrixXf J_GloveDifferenceMaxMinPCALimit = Eigen::MatrixXf::Zero(K, NUM_HAND_FINGER_PARAMS);
	Eigen::VectorXf e_GloveDifferenceMaxMinPCALimit = Eigen::VectorXf::Zero(K);

	Eigen::VectorXf Params_fingers = this->Hand_Params.tail(NUM_HAND_FINGER_PARAMS);
	Eigen::VectorXf Params_glove_fingers = this->Glove_params.tail(NUM_HAND_FINGER_PARAMS);

	//�������PCA�任
	Eigen::VectorXf Params_fingers_Pca = mHandModel->Hands_components.leftCols(K).transpose() * (Params_fingers - mHandModel->Hands_mean);
	Eigen::VectorXf Params_glove_fingers_Pca = mHandModel->Hands_components.leftCols(K).transpose() * (Params_glove_fingers - mHandModel->Hands_mean);

	Eigen::VectorXf Params_finger_PCA_Difference = Params_fingers_Pca - Params_glove_fingers_Pca;

	for (int i = 0; i < K; ++i)
	{
		//�ֱ����ÿһ��ά��
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
	linear_system.rhs.tail(NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_Difference_MAXMIN_PCA_weight*J_GloveDifferenceMaxMinPCALimit.transpose() * e_GloveDifferenceMaxMinPCALimit;
}

void Worker::GloveDifferenceVarPCALimit(LinearSystem& linear_system)
{
	int K = mHandModel->K_PCAcomponnet;

	Eigen::VectorXf Params_fingers = this->Hand_Params.tail(NUM_HAND_FINGER_PARAMS);
	Eigen::VectorXf Params_glove_fingers = this->Glove_params.tail(NUM_HAND_FINGER_PARAMS);

	//�������PCA�任
	Eigen::VectorXf Params_fingers_Pca = mHandModel->Hands_components.leftCols(K).transpose() * (Params_fingers - mHandModel->Hands_mean);
	Eigen::VectorXf Params_glove_fingers_Pca = mHandModel->Hands_components.leftCols(K).transpose() * (Params_glove_fingers - mHandModel->Hands_mean);

	Eigen::VectorXf Params_finger_PCA_Difference = Params_fingers_Pca - Params_glove_fingers_Pca;


	//��������Ÿ���
	Eigen::MatrixXf P = mHandModel->Hands_components.leftCols(K).transpose();

	Eigen::MatrixXf pose_Sigma = Eigen::MatrixXf::Identity(K, K);
	pose_Sigma.diagonal() = mHandModel->Glove_Difference_VARPCA;
	Eigen::MatrixXf Invpose_Sigma = pose_Sigma.inverse();

	Eigen::MatrixXf J_GloveDifferenceVarPCALimit = Invpose_Sigma * P;
	Eigen::VectorXf e_GloveDifferenceVarPCALimit = -1 * Invpose_Sigma*Params_finger_PCA_Difference;

	linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
		NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
		NUM_HAND_FINGER_PARAMS, NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_Difference_Var_PCA_weight*J_GloveDifferenceVarPCALimit.transpose() * J_GloveDifferenceVarPCALimit;
	linear_system.rhs.tail(NUM_HAND_FINGER_PARAMS) += setting->Hand_Pose_Difference_Var_PCA_weight*J_GloveDifferenceVarPCALimit.transpose() * e_GloveDifferenceVarPCALimit;
}

void Worker::TemporalLimit(LinearSystem& linear_system, bool first_order)
{
	if (temporal_Hand_params.size() == 2)
	{
		Eigen::MatrixXf J_Tem = Eigen::MatrixXf::Identity(NUM_HAND_FINGER_PARAMS, NUM_HAND_FINGER_PARAMS);
		Eigen::VectorXf e_Tem = Eigen::VectorXf::Zero(NUM_HAND_FINGER_PARAMS);

		if (first_order)
			e_Tem = temporal_Hand_params.back() - Hand_Params.tail(NUM_HAND_FINGER_PARAMS);
		else
			e_Tem = 2 * temporal_Hand_params.back() - temporal_Hand_params.front() - Hand_Params.tail(NUM_HAND_FINGER_PARAMS);

		if (first_order)
		{
			linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS, NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
				NUM_HAND_FINGER_PARAMS, NUM_HAND_FINGER_PARAMS) += setting->Temporal_finger_params_FirstOrder_weight*J_Tem.transpose()*J_Tem;
			linear_system.rhs.tail(NUM_HAND_FINGER_PARAMS) += setting->Temporal_finger_params_FirstOrder_weight*J_Tem.transpose()*e_Tem;
		}
		else
		{
			linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS, NUM_HAND_SHAPE_PARAMS + NUM_HAND_GLOBAL_PARAMS,
				NUM_HAND_FINGER_PARAMS, NUM_HAND_FINGER_PARAMS) += setting->Temporal_finger_params_SecondOorder_weight*J_Tem.transpose()*J_Tem;
			linear_system.rhs.tail(NUM_HAND_FINGER_PARAMS) += setting->Temporal_finger_params_SecondOorder_weight*J_Tem.transpose()*e_Tem;
		}
	}
}

void Worker::Object_TemporalLimit(LinearSystem& linear_system, bool first_order, int obj_idx)
{
	if (temporal_Object_params[obj_idx].size() == 2)
	{
		Eigen::MatrixXf J_Tem = Eigen::MatrixXf::Identity(NUM_OBJECT_PARAMS, NUM_OBJECT_PARAMS);
		Eigen::VectorXf e_Tem = Eigen::VectorXf::Zero(NUM_OBJECT_PARAMS);

		if (first_order)
			e_Tem = temporal_Object_params[obj_idx].back() - Object_params[obj_idx];
		else
			e_Tem = 2*temporal_Object_params[obj_idx].back() - temporal_Object_params[obj_idx].front() - Object_params[obj_idx];

		if (first_order)
		{
			linear_system.lhs += setting->Object_Temporal_firstOrder_weight*J_Tem.transpose()*J_Tem;
			linear_system.rhs += setting->Object_Temporal_firstOrder_weight*J_Tem.transpose()*e_Tem;
		}
		else
		{
			linear_system.lhs += setting->Object_Temporal_secondOrder_weight*J_Tem.transpose()*J_Tem;
			linear_system.rhs += setting->Object_Temporal_secondOrder_weight*J_Tem.transpose()*e_Tem;
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
				if (mHandModel->Collision_Judge_Matrix(i, j) == 1) //������ײ������Ϊ i �� j ��ײ����i ��Ҫ�ƶ��������� j �� i ��ײ��j��Ҫ�ƶ���
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

void Worker::Object_CollisionLimit(LinearSystem& linear_system, int obj_idx)
{
	return;
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
	linear_system.lhs += D;
}

void Worker::Object_Damping(LinearSystem& linear_system)
{
	Eigen::MatrixXf D = Eigen::MatrixXf::Identity(NUM_OBJECT_PARAMS, NUM_OBJECT_PARAMS);

	D(0, 0) = setting->Object_Trans_Damping;
	D(1, 1) = setting->Object_Trans_Damping;
	D(2, 2) = setting->Object_Trans_Damping;

	D(3, 3) = setting->Object_Rotate_Damping;
	D(4, 4) = setting->Object_Rotate_Damping;
	D(5, 5) = setting->Object_Rotate_Damping;

	linear_system.lhs += D;
}

void Worker::Hand_Object_Collision(LinearSystem& linear_system)
{
	vector<pair<int,Vector3>> hand_obj_col;

	for (int v_id = 0; v_id < mHandModel->Vertex_num; ++v_id)
	{
		for (int obj_idx = 0; obj_idx < mInteracted_Objects.size(); ++obj_idx)
		{
			Eigen::Vector3f p(mHandModel->V_Final(v_id, 0), mHandModel->V_Final(v_id, 1), mHandModel->V_Final(v_id, 2));
			if (mInteracted_Objects[obj_idx]->Is_inside(p))
			{
				Eigen::Vector3f cor = mInteracted_Objects[obj_idx]->FindTarget(p);
				hand_obj_col.emplace_back(make_pair(v_id, cor));
			}
		}
	}

	int collision_size = static_cast<int>(hand_obj_col.size());

	if (collision_size > 0)
	{
		Eigen::VectorXf e = Eigen::VectorXf::Zero(collision_size * 3);
		Eigen::MatrixXf J = Eigen::MatrixXf::Zero(collision_size * 3, NUM_HAND_POSE_PARAMS);
		Eigen::MatrixXf pose_jacob;

		for (int i = 0; i < collision_size; ++i)
		{
			int v_id = hand_obj_col[i].first;

			e(i * 3 + 0) = (hand_obj_col[i].second)(0) - mHandModel->V_Final(v_id, 0);
			e(i * 3 + 1) = (hand_obj_col[i].second)(1) - mHandModel->V_Final(v_id, 1);
			e(i * 3 + 2) = (hand_obj_col[i].second)(2) - mHandModel->V_Final(v_id, 2);

			mHandModel->Pose_jacobain(pose_jacob, v_id);

			J.block(i * 3, 0, 3, NUM_HAND_POSE_PARAMS) = pose_jacob;
		}

		linear_system.lhs.block(NUM_HAND_SHAPE_PARAMS, NUM_HAND_SHAPE_PARAMS,
			NUM_HAND_POSE_PARAMS, NUM_HAND_POSE_PARAMS) += setting->Hand_object_collision * J.transpose() * J;
		linear_system.rhs.tail(NUM_HAND_POSE_PARAMS) += setting->Hand_object_collision*J.transpose()*e;
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

	//��ͨ�����ֺ���������ɵ����ͼ�ϳ�һ����������ͼ���������ǰ���ϵ���кϲ�
	for(int obj_idx = 0;obj_idx<mInteracted_Objects.size();++obj_idx)
		mInteracted_Objects[obj_idx]->GenerateDepthAndSilhouette();

	mRendered_Images.setToZero();

	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			bool is_object = false;
			for (int obj_idx = 0; obj_idx < mInteracted_Objects.size(); ++obj_idx)
				is_object |= (mInteracted_Objects[obj_idx]->generatedSilhouette.at<uchar>(row, col) == 255);
			
			bool is_hand = (mHandModel->HandModel_binaryMap.at<uchar>(row, col) == 255);
			bool is_origin = (mImage_InputData->silhouette.at<uchar>(row, col) == 255);

			if (is_hand || is_object||is_origin)
			{
				ushort min_obj_depth = 10000;
				for (int obj_idx = 0; obj_idx < mInteracted_Objects.size(); ++obj_idx)
				{
					ushort obj_depth = mInteracted_Objects[obj_idx]->generatedDepth.at<ushort>(row, col);
					if (obj_depth != 0 && obj_depth < min_obj_depth)
						min_obj_depth = mInteracted_Objects[obj_idx]->generatedDepth.at<ushort>(row, col);
				}
				ushort hand_depth = mHandModel->HandModel_depthMap.at<ushort>(row, col);
				ushort origin_depth = mImage_InputData->depth.at<ushort>(row, col);

				if (is_hand && !is_object)
				{
					mRendered_Images.rendered_hand_silhouette.at<uchar>(row, col) = 255;
					mRendered_Images.rendered_hand_depth.at<ushort>(row, col) = hand_depth;

					count++;
					errors += abs(hand_depth - origin_depth) > MaxThreshold ? MaxThreshold : abs(hand_depth - origin_depth);
				}

				if (!is_hand && is_object)
				{
					mRendered_Images.rendered_object_silhouette.at<uchar>(row, col) = 255;
					mRendered_Images.rendered_object_depth.at<ushort>(row, col) = min_obj_depth;

					count++;
					errors += abs(min_obj_depth - origin_depth) > MaxThreshold ? MaxThreshold : abs(min_obj_depth - origin_depth);
				}

				if (is_hand && is_object)
				{
					if (min_obj_depth < hand_depth)
					{
						mRendered_Images.rendered_object_silhouette.at<uchar>(row, col) = 255;
						mRendered_Images.rendered_object_depth.at<ushort>(row, col) = min_obj_depth;

						count++;
						errors += abs(min_obj_depth - origin_depth) > MaxThreshold ? MaxThreshold : abs(min_obj_depth - origin_depth);
					}
					else
					{
						mRendered_Images.rendered_hand_silhouette.at<uchar>(row, col) = 255;
						mRendered_Images.rendered_hand_depth.at<ushort>(row, col) = hand_depth;

						count++;
						errors += abs(hand_depth - origin_depth) > MaxThreshold ? MaxThreshold : abs(hand_depth - origin_depth);
					}
				}

				if (!is_hand && !is_object)
				{
					count++;
					errors += origin_depth > MaxThreshold ? MaxThreshold : origin_depth;
				}
			}
		}
	}

	if (count <= 0)
		tracking_success = false;
	else
	{
		tracking_error = errors / count;
		tracking_success = tracking_error > setting->tracking_fail_threshold ? false : true;
	}
}