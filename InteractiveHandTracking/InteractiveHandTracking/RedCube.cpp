#include"Interacted_Object.h"

RedCube::RedCube(Camera* camera) :Interacted_Object(camera)
{
	mObject_attribute.type = redCube;
	mObject_attribute.color = Vector3(1.0f, 0.0f, 0.0f);
	mObject_attribute.length = 50.0f;
	mObject_attribute.CornerPoints = Eigen::Matrix<float, 8, 3>::Zero();
	{
		mObject_attribute.CornerPoints.row(0) = Eigen::RowVector3f(-25.0f, 25.0f, 25.0f);
		mObject_attribute.CornerPoints.row(1) = Eigen::RowVector3f(-25.0f, 25.0f, -25.0f);
		mObject_attribute.CornerPoints.row(2) = Eigen::RowVector3f(25.0f, 25.0f, -25.0f);
		mObject_attribute.CornerPoints.row(3) = Eigen::RowVector3f(25.0f, 25.0f, 25.0f);

		mObject_attribute.CornerPoints.row(4) = Eigen::RowVector3f(-25.0f, -25.0f, 25.0f);
		mObject_attribute.CornerPoints.row(5) = Eigen::RowVector3f(-25.0f, -25.0f, -25.0f);
		mObject_attribute.CornerPoints.row(6) = Eigen::RowVector3f(25.0f, -25.0f, -25.0f);
		mObject_attribute.CornerPoints.row(7) = Eigen::RowVector3f(25.0f, -25.0f, 25.0f);
	}
	this->GenerateOrLoadPointsAndNormal();  //加载最初的点线面和法线

											//初始化网格参数
	Final_Vertices.assign(init_Vertices.begin(), init_Vertices.end());
	Final_Normal.assign(init_Normal.begin(), init_Normal.end());


	Visible_2D.reserve(init_Vertices.size());
	Visible_3D.reserve(init_Vertices.size());
	//初始化参数设为0
	this->Update(object_params);
}

void RedCube::GenerateOrLoadPointsAndNormal()
{
	Face_idx.clear();
	init_Vertices.clear();
	init_Normal.clear();

	int rows = 15;
	int cols = 15;

	float length = mObject_attribute.length;
	//6个面分别建立

	//首先是顶面
	{
		int idx_start = init_Vertices.size();
		Vector3 start_point = Vector3(-25.0f, 25.0f, -25.0f);
		Vector3 norm = Vector3(0, 1, 0);

		float y = start_point.y();
		for (int row = 0; row <= rows; ++row)
		{
			float z = start_point.z() + (float)row/rows * length;
			for (int col = 0; col <= cols; ++col)
			{
				float x = start_point.x() + (float)col/cols * length;
				init_Vertices.emplace_back(Vector3(x, y, z));
				init_Normal.push_back(norm);
			}
		}

		//再建立引索
		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				int p1 = idx_start + row * (cols+1) + col;
				int p2 = p1 + cols + 1;

				Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
				Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
			}
		}
	}

	//然后是底面
	{
		int idx_start = init_Vertices.size();
		Vector3 start_point = Vector3(-25.0f, -25.0f, -25.0f);
		Vector3 norm = Vector3(0, -1, 0);

		float y = start_point.y();
		for (int row = 0; row <= rows; ++row)
		{
			float z = start_point.z() + (float)row/rows * length;
			for (int col = 0; col <= cols; ++col)
			{
				float x = start_point.x() + (float)col/cols * length;

				init_Vertices.emplace_back(Vector3(x, y, z));
				init_Normal.push_back(norm);
			}
		}

		//再建立引索
		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				int p1 = idx_start + row * (cols + 1) + col;
				int p2 = p1 + cols + 1;

				Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
				Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
			}
		}
	}

	//然后是前面
	{
		int idx_start = init_Vertices.size();
		Vector3 start_point = Vector3(-25.0f, -25.0f, 25.0f);
		Vector3 norm = Vector3(0, 0, 1);

		float z = start_point.z();
		for (int row = 0; row <= rows; ++row)
		{
			float y = start_point.y() + (float)row/rows * length;
			for (int col = 0; col <= cols; ++col)
			{
				float x = start_point.x() + (float)col/cols * length;

				init_Vertices.emplace_back(Vector3(x, y, z));
				init_Normal.push_back(norm);
			}
		}

		//再建立引索
		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				int p1 = idx_start + row * (cols + 1) + col;
				int p2 = p1 + cols + 1;

				Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
				Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
			}
		}
	}

	//然后是后面
	{
		int idx_start = init_Vertices.size();
		Vector3 start_point = Vector3(-25.0f, -25.0f, -25.0f);
		Vector3 norm = Vector3(0, 0, -1);

		float z = start_point.z();
		for (int row = 0; row <= rows; ++row)
		{
			float y = start_point.y() + (float)row/rows * length;
			for (int col = 0; col <= cols; ++col)
			{
				float x = start_point.x() + (float)col/cols * length;

				init_Vertices.emplace_back(Vector3(x, y, z));
				init_Normal.push_back(norm);
			}
		}

		//再建立引索
		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				int p1 = idx_start + row * (cols + 1) + col;
				int p2 = p1 + cols + 1;

				Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
				Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
			}
		}
	}

	//然后是左面
	{
		int idx_start = init_Vertices.size();
		Vector3 start_point = Vector3(-25.0f, -25.0f, -25.0f);
		Vector3 norm = Vector3(-1, 0, 0);

		float x = start_point.x();
		for (int row = 0; row <= rows; ++row)
		{
			float y = start_point.y() + (float)row/rows * length;
			for (int col = 0; col <= cols; ++col)
			{
				float z = start_point.z() + (float)col/cols * length;

				init_Vertices.emplace_back(Vector3(x, y, z));
				init_Normal.push_back(norm);
			}
		}

		//再建立引索
		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				int p1 = idx_start + row * (cols + 1) + col;
				int p2 = p1 + cols + 1;

				Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
				Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
			}
		}
	}

	//最后是右面
	{
		int idx_start = init_Vertices.size();
		Vector3 start_point = Vector3(25.0f, -25.0f, -25.0f);
		Vector3 norm = Vector3(1, 0, 0);

		float x = start_point.x();
		for (int row = 0; row <= rows; ++row)
		{
			float y = start_point.y() + (float)row/rows * length;
			for (int col = 0; col <= cols; ++col)
			{
				float z = start_point.z() + (float)col/cols * length;

				init_Vertices.emplace_back(Vector3(x, y, z));
				init_Normal.push_back(norm);
			}
		}

		//再建立引索
		for (int row = 0; row < rows; ++row)
		{
			for (int col = 0; col < cols; ++col)
			{
				int p1 = idx_start + row * (cols + 1) + col;
				int p2 = p1 + cols + 1;

				Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
				Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
			}
		}
	}

	cout << "顶点数量为：" << init_Vertices.size() << endl;
}

void RedCube::GenerateDepthAndSilhouette()
{
	Vector3 center = object_params.head(3);
	this->generatedDepth.setTo(0);
	this->generatedSilhouette.setTo(0);

	if (center.z() < 200) return;

	int cols = generatedDepth.cols;
	int rows = generatedDepth.rows;

	int Face_num = Face_idx.size();
	for (int faceidx = 0; faceidx < Face_num; ++faceidx)
	{
		int a_idx = Face_idx[faceidx](0);
		int b_idx = Face_idx[faceidx](1);
		int c_idx = Face_idx[faceidx](2);

		Eigen::Vector2f a_2D = mCamera->world_to_image(Final_Vertices[a_idx]);
		Eigen::Vector2f b_2D = mCamera->world_to_image(Final_Vertices[b_idx]);
		Eigen::Vector2f c_2D = mCamera->world_to_image(Final_Vertices[c_idx]);

		int x_min = min(min(a_2D.x(), b_2D.x()), c_2D.x());
		int y_min = min(min(a_2D.y(), b_2D.y()), c_2D.y());
		int x_max = max(max(a_2D.x(), b_2D.x()), c_2D.x());
		int y_max = max(max(a_2D.y(), b_2D.y()), c_2D.y());

		//插值深度，并且进行判断
		float alpha0 = 0, alpha1 = 0;
		float depthA = 0, depthB = 0, depthC = 0;

		float d0 = 0, d1 = 0, d2 = 0;

		Eigen::Vector2f vector_aTob = b_2D - a_2D;
		Eigen::Vector2f vector_aToc = c_2D - a_2D;

		depthA = Final_Vertices[a_idx](2);
		depthB = Final_Vertices[b_idx](2);
		depthC = Final_Vertices[c_idx](2);

		for (int y = y_min; y <= y_max; ++y)
		{
			for (int x = x_min; x <= x_max; ++x)
			{
				if (x >= 0 && x < cols && y >= 0 && y < rows)
				{
					int a = (b_2D.x() - a_2D.x()) * (y - a_2D.y()) - (b_2D.y() - a_2D.y()) * (x - a_2D.x());
					int b = (c_2D.x() - b_2D.x()) * (y - b_2D.y()) - (c_2D.y() - b_2D.y()) * (x - b_2D.x());
					int c = (a_2D.x() - c_2D.x()) * (y - c_2D.y()) - (a_2D.y() - c_2D.y()) * (x - c_2D.x());

					if ((a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0))
					{
						generatedSilhouette.at<uchar>(y, x) = 255;

						Eigen::Vector2f p(x, y);
						Eigen::Vector2f vector_aTop = p - a_2D;
						Eigen::Vector2f vector_bTop = p - b_2D;

						//叉乘计算面积
						float S_abc = vector_aTob.x() * vector_aToc.y() - vector_aToc.x() * vector_aTob.y();
						float S_abp = vector_aTob.x() * vector_aTop.y() - vector_aTop.x() * vector_aTob.y();
						float S_apc = vector_aTop.x() * vector_aToc.y() - vector_aToc.x() * vector_aTop.y();

						if (S_abc != 0) {
							alpha1 = S_abp / S_abc;
							alpha0 = S_apc / S_abc;
						}
						else {
							//说明这些点共线

							//如果 vector_aTob 和vector_aTop方向相同
							if (vector_aTob.dot(vector_aTop) >= 0)
							{
								alpha1 = 0;
								if (vector_aTob.y() != 0) {
									alpha0 = (vector_aTop.y()) / (vector_aTob.y());
								}
								else { alpha0 = (vector_aTop.x()) / (vector_aTob.x()); }
							}
							else
							{
								alpha0 = 0;
								if (vector_aToc.y() != 0) {
									alpha1 = (vector_aTop.y()) / (vector_aToc.y());
								}
								else { alpha1 = (vector_aTop.x()) / (vector_aToc.x()); }
							}
						}

						//重心坐标系相关知识
						float depth = depthA + alpha0*(depthB - depthA) + alpha1*(depthC - depthA);
						ushort v = generatedDepth.at<ushort>(y, x);
						if (v != 0) {
							generatedDepth.at<ushort>(y, x) = min(v, (ushort)depth);
						}
						else {
							generatedDepth.at<ushort>(y, x) = (ushort)depth;
						}
					}
				}
			}
		}
	}
}

void RedCube::ShowDepth()
{
	cv::Mat tmp_depth = generatedDepth.clone();
	tmp_depth.setTo(0, generatedSilhouette == 0);
	double max, min;
	cv::minMaxIdx(tmp_depth, &min, &max);
	cv::Mat normal_map;
	tmp_depth.convertTo(normal_map, CV_8U, 255.0 / (max - min), -min * 255.0 / (max - min));  //灰度拉升
	cv::Mat junheng_Map;
	cv::equalizeHist(normal_map, junheng_Map);   //直方图均衡提高对比度
	cv::Mat color_map;
	cv::applyColorMap(junheng_Map, color_map, cv::COLORMAP_COOL);  //伪彩色
	color_map.setTo(cv::Scalar(255, 255, 255), generatedSilhouette == 0);  //手部以外其他地方置零
	cv::flip(color_map, color_map, 0);
	cv::imshow("RedCube生成的深度图", color_map);
}

void RedCube::Update(const Eigen::VectorXf& params)
{
	object_params = params;
	this->UpdateVerticesAndNormal();
}


void RedCube::UpdateVerticesAndNormal()
{
	float width = mCamera->width();
	float heigh = mCamera->height();

	Visible_3D.points.clear();
	Visible_2D.clear();

	Eigen::Matrix4f TransMatrix = Eigen::Matrix4f::Identity();
	TransMatrix(0, 3) = object_params(0);
	TransMatrix(1, 3) = object_params(1);
	TransMatrix(2, 3) = object_params(2);

	Eigen::Matrix4f RotateMatrix = Eigen::Matrix4f::Identity();
	RotateMatrix.block(0, 0, 3, 3) = EularToRotateMatrix(object_params(3), object_params(4), object_params(5));

	Eigen::Matrix4f updataMatrix = TransMatrix * RotateMatrix;

	for (size_t i = 0; i < init_Vertices.size(); ++i)
	{
		Eigen::Vector4f tmp(init_Vertices[i](0), init_Vertices[i](1), init_Vertices[i](2), 1);
		Final_Vertices[i] = (updataMatrix * tmp).head(3);

		Eigen::Vector4f norm(init_Normal[i](0), init_Normal[i](1), init_Normal[i](2), 0);
		Final_Normal[i] = (updataMatrix * norm).head(3);

		if (Final_Normal[i].z() < 0)
		{
			Vector2 p_2D = mCamera->world_to_image(Final_Vertices[i]);
			if (p_2D.x() >= 0 && p_2D.x() < width && p_2D.y() >= 0 && p_2D.y() < heigh)
			{
				Visible_2D.emplace_back(make_pair(Vector3(p_2D(0), p_2D(1), Final_Vertices[i](2)), i));

				pcl::PointNormal p_3D;
				p_3D.x = Final_Vertices[i].x();
				p_3D.y = Final_Vertices[i].y();
				p_3D.z = Final_Vertices[i].z();

				p_3D.normal_x = Final_Normal[i].x();
				p_3D.normal_y = Final_Normal[i].y();
				p_3D.normal_z = Final_Normal[i].z();

				Visible_3D.points.emplace_back(p_3D);
			}
		}
	}

	//更新角点以及局部坐标系
	for (int i = 0; i < 8; ++i)
	{
		Eigen::Vector4f tmp(mObject_attribute.CornerPoints(i, 0), mObject_attribute.CornerPoints(i, 1), mObject_attribute.CornerPoints(i, 2), 1);
		FinalCornerPoints.row(i) = ((updataMatrix * tmp).head(3)).transpose();
	}

	Eigen::Vector4f X_tmp(1, 0, 0, 0);
	Eigen::Vector4f Y_tmp(0, 1, 0, 0);
	Eigen::Vector4f Z_tmp(0, 0, 1, 0);

	Coordinate.col(0) = (updataMatrix * X_tmp).head(3);
	Coordinate.col(1) = (updataMatrix * Y_tmp).head(3);
	Coordinate.col(2) = (updataMatrix * Z_tmp).head(3);
}


bool RedCube::Is_inside(const Eigen::VectorXf& p)
{
	float Half_Len = mObject_attribute.length / 2.0f;

	Vector3 C = Vector3(object_params(0), object_params(1), object_params(2));
	Vector3 CP = p - C;

	//计算投影
	float L_OX = CP.dot(Coordinate.col(0));
	float L_OY = CP.dot(Coordinate.col(1));
	float L_OZ = CP.dot(Coordinate.col(2));

	return (abs(L_OX) < Half_Len) && (abs(L_OY) < Half_Len) && (abs(L_OZ) < Half_Len);
}

Eigen::VectorXf RedCube::FindTarget(const Eigen::VectorXf& p)
{
	float Half_Len = mObject_attribute.length / 2.0f;

	Vector3 C = Vector3(object_params(0), object_params(1), object_params(2));
	Vector3 CP = p - C;

	//计算投影
	float L_OX = CP.dot(Coordinate.col(0));
	float L_OY = CP.dot(Coordinate.col(1));
	float L_OZ = CP.dot(Coordinate.col(2));

	Vector3 target;
	//找到最大的投影，因为投影越大，说明距离中心越远，距离表面就越近
	if (abs(L_OX) > abs(L_OY) && abs(L_OX) > abs(L_OZ))
	{
		Vector3 dir = L_OX / abs(L_OX) * Coordinate.col(0);
		float scale = Half_Len - abs(L_OX);

		target =  p + scale * dir;
	}
	else if (abs(L_OY) > abs(L_OX) && abs(L_OY) > abs(L_OZ))
	{
		Vector3 dir = L_OY / abs(L_OY) * Coordinate.col(1);
		float scale = Half_Len - abs(L_OY);

		target = p + scale * dir;
	}
	else
	{
		Vector3 dir = L_OZ / abs(L_OZ) * Coordinate.col(2);
		float scale = Half_Len - abs(L_OZ);

		target = p + scale * dir;
	}

	return target;
}

Eigen::VectorXf RedCube::FindTouchPoint(const Eigen::VectorXf& p)
{
	if (Is_inside(p))
		return FindTarget(p);
	else
	{
		Eigen::Matrix4f T_local = Eigen::Matrix4f::Identity();
		T_local.block(0, 0, 3, 3) = Coordinate;
		T_local(0, 3) = object_params(0);
		T_local(1, 3) = object_params(1);
		T_local(2, 3) = object_params(2);

		Eigen::Vector4f p_tmp = Eigen::Vector4f(p(0), p(1), p(2), 1);
		Eigen::Vector3f p_local = (T_local.inverse() * p_tmp).head(3);

		Eigen::Vector3f diff;
		diff << max(abs(p_local(0)) - mObject_attribute.length / 2.0f,0.0f),
			max(abs(p_local(1)) - mObject_attribute.length / 2.0f,0.0f),
			max(abs(p_local(2)) - mObject_attribute.length / 2.0f,0.0f);

		Eigen::Vector3f dir;
		dir(0) = p_local(0) > 0 ? -1 : 1;
		dir(1) = p_local(1) > 0 ? -1 : 1;
		dir(2) = p_local(2) > 0 ? -1 : 1;

		Eigen::Vector4f target_local;
		target_local(0) = p_local(0) + dir(0) * diff(0);
		target_local(1) = p_local(1) + dir(1) * diff(1);
		target_local(2) = p_local(2) + dir(2) * diff(2);
		target_local(1) = 1;

		return (T_local * target_local).head(3);
	}

	return Eigen::Vector3f::Zero();
}