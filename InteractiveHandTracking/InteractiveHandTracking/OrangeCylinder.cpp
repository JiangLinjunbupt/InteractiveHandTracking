#include"Interacted_Object.h"


GreenCylinder::GreenCylinder(Camera* camera):Interacted_Object(camera)
{
	mObject_attribute.type = greenCylinder;
	mObject_attribute.color = Vector3(1.0f, 0.5f, 0);
	mObject_attribute.radius = 12.5f;
	mObject_attribute.height = 75.0f;

	this->GenerateOrLoadPointsAndNormal();

	Final_Vertices.assign(init_Vertices.begin(), init_Vertices.end());
	Final_Normal.assign(init_Normal.begin(), init_Normal.end());


	Visible_2D.reserve(init_Vertices.size());
	Visible_3D.reserve(init_Vertices.size());
	//初始化参数设为0
	this->Update(object_params);
}

void GreenCylinder::GenerateOrLoadPointsAndNormal()
{
	Face_idx.clear();
	init_Vertices.clear();
	init_Normal.clear();

	int numCir = 10;
	int numLine = 20;

	int numHeight = 20;


	//先生成上顶面
	{
		Vector3 n = Vector3(0, 1, 0);
		float y = mObject_attribute.height / 2.0f;

		float margin_Cir = mObject_attribute.radius / numCir;
		float margin_Lin = 2.0f * M_PI / numLine;

		for (int i = 0; i <= numCir; ++i)
		{
			float radius_tmp = i * margin_Cir;

			for (int j = 0; j <= numLine; ++j)
			{
				float x = radius_tmp * cos(j * margin_Lin);
				float z = radius_tmp * sin(j * margin_Lin);

				Vector3 v = Vector3(x, y, z);
				init_Vertices.push_back(v);
				init_Normal.push_back(n);
			}
		}

		for (int i = 0; i < numCir; ++i)
		{
			for (int j = 0; j < numLine; ++j)
			{
				int p1 = i * (numLine + 1) + j;
				int p2 = p1 + numLine + 1;

				Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
				Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
			}
		}
	}

	//再生成下底面
	{
		int idx_start = init_Vertices.size();

		Vector3 n = Vector3(0, -1, 0);
		float y = mObject_attribute.height / (-2.0f);

		float margin_Cir = mObject_attribute.radius / numCir;
		float margin_Lin = 2.0f * M_PI / numLine;

		for (int i = 0; i <= numCir; ++i)
		{
			float radius_tmp = i * margin_Cir;

			for (int j = 0; j <= numLine; ++j)
			{
				float x = radius_tmp * cos(j * margin_Lin);
				float z = radius_tmp * sin(j * margin_Lin);

				Vector3 v = Vector3(x, y, z);
				init_Vertices.push_back(v);
				init_Normal.push_back(n);
			}
		}

		for (int i = 0; i < numCir; ++i)
		{
			for (int j = 0; j < numLine; ++j)
			{
				int p1 = idx_start + i * (numLine + 1) + j;
				int p2 = p1 + numLine + 1;

				Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
				Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
			}
		}
	}

	//最后生成圆柱侧面
	{
		int idx_start = init_Vertices.size();
		float margin_height = mObject_attribute.height / numHeight;
		float margin_Line = 2.0f * M_PI / numLine;

		for (int i = 0; i <= numHeight; ++i)
		{
			float y = mObject_attribute.height / 2.0f - i * margin_height;

			for (int j = 0; j <= numLine; ++j)
			{
				float x = mObject_attribute.radius * cos(j * margin_Line);
				float z = mObject_attribute.radius * sin(j * margin_Line);

				Vector3 v = Vector3(x, y, z);
				Vector3 n = Vector3(x, 0, z);
				n.normalize();

				init_Vertices.push_back(v);
				init_Normal.push_back(n);
			}
		}

		for (int i = 0; i < numHeight; ++i)
		{
			for (int j = 0; j < numLine; ++j)
			{
				int p1 = idx_start + i * (numLine + 1) + j;
				int p2 = p1 + numLine + 1;

				Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
				Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
			}
		}
	}
}

void GreenCylinder::GenerateDepthAndSilhouette()
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

void GreenCylinder::ShowDepth()
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

void GreenCylinder::Update(const Eigen::VectorXf& params)
{
	object_params = params;
	this->UpdateVerticesAndNormal();
}

void GreenCylinder::UpdateVerticesAndNormal()
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

	Eigen::Vector4f X_tmp(1, 0, 0, 0);
	Eigen::Vector4f Y_tmp(0, 1, 0, 0);
	Eigen::Vector4f Z_tmp(0, 0, 1, 0);

	Coordinate.col(0) = (updataMatrix * X_tmp).head(3);
	Coordinate.col(1) = (updataMatrix * Y_tmp).head(3);
	Coordinate.col(2) = (updataMatrix * Z_tmp).head(3);

	T_local.setIdentity();
	T_local.block(0, 0, 3, 3) = Coordinate;
	T_local(0, 3) = object_params(0);
	T_local(1, 3) = object_params(1);
	T_local(2, 3) = object_params(2);

	T_local_inverse = T_local.inverse();
}

Eigen::MatrixXf GreenCylinder::GetObjectTransMatrix()
{
	Eigen::Matrix4f TransMatrix = Eigen::Matrix4f::Identity();
	TransMatrix(0, 3) = object_params(0);
	TransMatrix(1, 3) = object_params(1);
	TransMatrix(2, 3) = object_params(2);

	Eigen::Matrix4f RotateMatrix = Eigen::Matrix4f::Identity();
	RotateMatrix.block(0, 0, 3, 3) = EularToRotateMatrix(object_params(3), object_params(4), object_params(5));

	return TransMatrix * RotateMatrix;
}

float GreenCylinder::SDF(const Eigen::Vector3f& p)
{
	Eigen::Vector4f p_tmp(p(0), p(1), p(2), 1);
	Vector3 p_local = (T_local_inverse *p_tmp).head(3);

	float r = mObject_attribute.radius;
	float h_half = mObject_attribute.height / 2.0f;

	float l = sqrt(p_local.x() * p_local.x() + p_local.z() * p_local.z());

	Vector2 p_2D = Vector2(l, abs(p_local.y()));
	Vector2 diff = p_2D - Vector2(r, h_half);

	return min(max(diff.x(), diff.y()), 0.0f) + sqrt(
		max(diff.x(), 0.0f) * max(diff.x(), 0.0f) +
		max(diff.y(), 0.0f) * max(diff.y(), 0.0f)
	);
}

bool GreenCylinder::Is_inside(const Eigen::VectorXf& p)
{
	if (this->SDF(p) < 0)
		return true;
	else
		return false;
}

//这个函数的前提是：p点在圆柱体内部
Eigen::VectorXf GreenCylinder::FindTarget(const Eigen::VectorXf& p)
{
	Eigen::Vector4f p_tmp(p(0), p(1), p(2), 1);
	Vector3 p_local = (T_local_inverse *p_tmp).head(3);

	float r = mObject_attribute.radius;
	float h_half = mObject_attribute.height / 2.0f;

	float L1 = r - sqrt(p_local.x() * p_local.x() + p_local.z() * p_local.z());
	float L2 = h_half - abs(p_local.y());
	//现在需要决定的是距离圆柱侧面近还是距离两个顶面近

	Vector2 dir(p_local.x(), p_local.z());
	dir.normalize();

	int y_flag = (p_local.y() > 0 ? 1 : -1);

	Eigen::Vector4f target_local;
	if (L1 < L2)
	{
		target_local << dir.x() * r,
			p_local.y(),
			dir.y() * r,
			1;
	}
	else
	{
		target_local << p_local.x(),
			h_half * y_flag,
			p_local.z(),
			1;
	}

	return (T_local * target_local).head(3);
}

Eigen::VectorXf GreenCylinder::FindTouchPoint(const Eigen::VectorXf& p)
{
	if (Is_inside(p))
		return FindTarget(p);
	else
	{
		Eigen::Vector4f p_tmp(p(0), p(1), p(2), 1);
		Vector3 p_local = (T_local_inverse *p_tmp).head(3);

		float r = mObject_attribute.radius;
		float h_half = mObject_attribute.height / 2.0f;

		Vector2 dir(p_local.x(), p_local.z());
		dir.normalize();

		int y_flag = (p_local.y() > 0 ? 1 : -1);

		float l = sqrt(p_local.x() * p_local.x() + p_local.z() * p_local.z());

		if (l >= r && abs(p_local.y()) >= h_half)
		{
			return (T_local * Eigen::Vector4f(dir.x() * r, h_half * y_flag, dir.y() *r, 1)).head(3);
		}

		if (l <= r && abs(p_local.y()) >= h_half)
		{
			return (T_local * Eigen::Vector4f(p_local.x(), h_half * y_flag, p_local.z(), 1)).head(3);
		}

		if (l >= r && abs(p_local.y()) <= h_half)
		{
			return (T_local * Eigen::Vector4f(dir.x()*r, p_local.y(), dir.y()*r, 1)).head(3);
		}
	}

	return Vector3::Zero();
}

