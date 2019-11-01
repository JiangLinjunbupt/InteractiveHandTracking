#include"Worker.h"

Worker::Worker(Interacted_Object* _interacted_Object, Camera* _camera) :mInteracted_Object(_interacted_Object), mCamera(_camera)
{
	tracking_error = 0.0f;
	tracking_success = false;
	itr = 0;
	total_itr = 0;
	while (!temporal_Object_params.empty()) temporal_Object_params.pop();
	Object_params_num = 6;
	mImage_InputData = nullptr;
	Object_params = Eigen::VectorXf::Zero(Object_params_num);
	correspond.clear();

	mInteracted_Object->Update(Object_params);
}

void Worker::Tracking(const Eigen::VectorXf& startData, Image_InputData& imageData, bool previous_success, const Eigen::VectorXf& previous_best_estimation)
{
	tracking_success = false;
	itr = 0;

	SetInputData(startData, imageData);
	SetTemporalInfo(previous_success, previous_best_estimation);

	for (; itr < setting->max_itr; ++itr)
	{
		FindObjectCorrespond();
		One_tracking();
	}

	Evaluation();
}

void Worker::SetInputData(const Eigen::VectorXf& startData, Image_InputData& imageData)
{
	Object_params = startData;
	mImage_InputData = &imageData;

	mInteracted_Object->Update(Object_params);
}

void Worker::SetTemporalInfo(bool previous_success, const Eigen::VectorXf& previous_best_estimation)
{
	if (previous_success)
	{
		if (temporal_Object_params.size() == 2) {
			temporal_Object_params.pop();
			temporal_Object_params.push(previous_best_estimation);
		}
		else
		{
			temporal_Object_params.push(previous_best_estimation);
		}
	}
	else
	{
		while (!temporal_Object_params.empty())
			temporal_Object_params.pop();
	}
}

void Worker::FindObjectCorrespond()
{
	correspond.clear();

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
		correspond.resize(NumPointCloud_sensor);
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
			correspond[i].pointcloud = p;
			

			Eigen::Vector3f p_cor = Eigen::Vector3f(object_visible_cloud.points[k_indices[0]].x,
				object_visible_cloud.points[k_indices[0]].y,
				object_visible_cloud.points[k_indices[0]].z);

			correspond[i].correspond = p_cor;
			correspond[i].correspond_idx = visible_idx[k_indices[0]];

			float distance = (p_cor - p).norm();

			if (distance > 50)
				correspond[i].is_match = false;
			else
				correspond[i].is_match = true;
		}
	}

	num_matched_correspond = 0;

	for (std::vector<DataAndCorrespond>::iterator itr = correspond.begin(); itr != correspond.end(); ++itr)
	{
		if (itr->is_match)
			++num_matched_correspond;
	}
}

void Worker::One_tracking()
{
	//初始化求解相关
	LinearSystem linear_system;
	linear_system.lhs = Eigen::MatrixXf::Zero(Object_params_num, Object_params_num);
	linear_system.rhs = Eigen::VectorXf::Zero(Object_params_num);

	this->Fitting3D(linear_system);
	this->Fitting2D(linear_system);
	this->TemporalLimit(linear_system, true);
	this->TemporalLimit(linear_system, false);
	this->Damping(linear_system);

	Eigen::VectorXf solution = this->Solver(linear_system);

	for (int i = 0; i < Object_params_num; ++i)
		Object_params[i] += solution[i];

	mInteracted_Object->Update(Object_params);
}

void Worker::Fitting3D(LinearSystem& linear_system)
{
	int NumofCorrespond = num_matched_correspond;
	Eigen::VectorXf e = Eigen::VectorXf::Zero(NumofCorrespond * 3);
	Eigen::MatrixXf J = Eigen::MatrixXf::Zero(NumofCorrespond * 3, Object_params_num);

	//临时变量
	Eigen::MatrixXf tmp_jacob;
	int count = 0;

	for (int i = 0; i < correspond.size(); ++i)
	{
		if (correspond[i].is_match)
		{
			int v_id = correspond[i].correspond_idx;

			e(count * 3 + 0) = correspond[i].pointcloud(0) - correspond[i].correspond(0);
			e(count * 3 + 1) = correspond[i].pointcloud(1) - correspond[i].correspond(1);
			e(count * 3 + 2) = correspond[i].pointcloud(2) - correspond[i].correspond(2);

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
			J.block(count * 3, 0, 3, Object_params_num) = tmp_jacob;
			++count;
		}
	}
	//std::cout << "--------> 第  " << itr << "  次迭代的误差为  ： " << total_error << std::endl;

	Eigen::MatrixXf JtJ = J.transpose()*J;
	Eigen::VectorXf JTe = J.transpose()*e;

	linear_system.lhs += setting->Object_fitting_3D*JtJ;
	linear_system.rhs += setting->Object_fitting_3D*JTe;
}

void Worker::Fitting2D(LinearSystem& linear_system)
{
	vector<pair<Eigen::Matrix2Xf, Eigen::Vector2f>> JacobVector_2D;

	int width = mCamera->width();
	int height = mCamera->height();

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
		Eigen::MatrixXf J_2D = Eigen::MatrixXf::Zero(2 * size, Object_params_num);
		Eigen::VectorXf e_2D = Eigen::VectorXf::Zero(2 * size);

		for (int i = 0; i < size; ++i)
		{
			J_2D.block(i * 2, 0, 2, Object_params_num) = JacobVector_2D[i].first;
			e_2D.segment(i * 2, 2) = JacobVector_2D[i].second;
		}

		linear_system.lhs += setting->Object_fitting_2D * J_2D.transpose() * J_2D;
		linear_system.rhs += setting->Object_fitting_2D * J_2D.transpose() * e_2D;
	}
}

void Worker::TemporalLimit(LinearSystem& linear_system, bool first_order)
{
	Eigen::MatrixXf J_Tem = Eigen::MatrixXf::Zero(Object_params_num, Object_params_num);
	Eigen::VectorXf e_Tem = Eigen::VectorXf::Zero(Object_params_num);

	Eigen::MatrixXf joint_jacob;

	if (temporal_Object_params.size() == 2)
	{
		for (int i = 0; i < Object_params_num; ++i)
		{
			int index = i;

			if (first_order)
			{
				e_Tem(i) = temporal_Object_params.back()(i) - Object_params(index);
			}
			else
			{
				e_Tem(i) = 2 * temporal_Object_params.back()(i) - temporal_Object_params.front()(i) - Object_params(index);
			}
		}
	}

	if (first_order)
	{
		linear_system.lhs.block(0, 0, Object_params_num, Object_params_num) += setting->Object_Temporal_firstOrder_weight*J_Tem.transpose()*J_Tem;
		linear_system.rhs.tail(Object_params_num) += setting->Object_Temporal_firstOrder_weight*J_Tem.transpose()*e_Tem;
	}
	else
	{
		linear_system.lhs.block(0, 0, Object_params_num, Object_params_num) += setting->Object_Temporal_secondOrder_weight*J_Tem.transpose()*J_Tem;
		linear_system.rhs.tail(Object_params_num) += setting->Object_Temporal_secondOrder_weight*J_Tem.transpose()*e_Tem;
	}
}

void Worker::Damping(LinearSystem& linear_system)
{
	Eigen::MatrixXf D = Eigen::MatrixXf::Identity(Object_params_num, Object_params_num);

	D(0, 0) = setting->Object_Trans_Damping;
	D(1, 1) = setting->Object_Trans_Damping;
	D(2, 2) = setting->Object_Trans_Damping;

	D(3, 3) = setting->Object_Rotate_Damping;
	D(4, 4) = setting->Object_Rotate_Damping;
	D(5, 5) = setting->Object_Rotate_Damping;

	linear_system.lhs += D;
}

Eigen::VectorXf Worker::Solver(LinearSystem& linear_system)
{
	///--- Solve for update dt = (J^T * J + D)^-1 * J^T * r

	//http://eigen.tuxfamily.org/dox/group__LeastSquares.html  for linear least square problem
	Eigen::VectorXf solution = linear_system.lhs.colPivHouseholderQr().solve(linear_system.rhs);

	///--- Check for NaN
	for (int i = 0; i<solution.size(); i++) {
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
	mInteracted_Object->GenerateDepthAndSilhouette();

	//然后在渲染得到的手模轮廓范围内比较
	int rows = mCamera->height();
	int cols = mCamera->width();

	float errors = 0.0f;
	int count = 0;
	float MaxThreshold = 100.0f;

	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			if (mInteracted_Object->generatedSilhouette.at<uchar>(row, col) == 255)
			{
				count++;
				float tmp_error = abs(mInteracted_Object->generatedDepth.at<ushort>(row, col) - mImage_InputData->depth.at<ushort>(row, col));
				errors += tmp_error > MaxThreshold ? MaxThreshold : tmp_error;
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