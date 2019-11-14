#include"Interacted_Object.h"


YellowSphere::YellowSphere(Camera* camera) : Interacted_Object(camera)
{
	mObject_attribute.type = yellowSphere;
	mObject_attribute.color = Vector3(1.0f, 1.0f, 0.0f);
	mObject_attribute.radius = 25.0f;
	this->GenerateOrLoadPointsAndNormal();  //��������ĵ�����ͷ���

	//��ʼ���������
	Final_Vertices.assign(init_Vertices.begin(),init_Vertices.end());
	Final_Normal.assign(init_Normal.begin(), init_Normal.end());

	Visible_2D.reserve(init_Vertices.size());
	Visible_3D.reserve(init_Vertices.size());
	//��ʼ��������Ϊ0
	this->Update(object_params);
}

void YellowSphere::GenerateOrLoadPointsAndNormal()
{
	float radius = mObject_attribute.radius;
	init_Vertices.clear();
	init_Normal.clear();
	Face_idx.clear();

	int numCir = 30;
	float margin_Cir = M_PI / numCir;

	int numLin = 30;
	float margin_Lin = 2 * M_PI / numLin;

	//�����ɵ�
	for (int i = 0; i <= numCir; ++i) {
		float y = radius * cos(i * margin_Cir);
		float radius_tmp = radius * sin(i * margin_Cir);
		for (int j = 0; j <= numLin; ++j)
		{
			//����λ��
			float x = radius_tmp * cos(margin_Lin * j);
			float z = radius_tmp * sin(margin_Lin * j);

			Vector3 v = Vector3(x, y, z);
			Vector3 n = Vector3(x / radius, y / radius, z / radius);

			init_Vertices.push_back(v);
			init_Normal.push_back(n);
		}
	}

	//����������
	for (int i = 0; i < numCir; ++i) {
		for (int j = 0; j < numLin; ++j)
		{
			//�����������
			int p1 = i * (numLin+1) + j;
			int p2 = p1 + numLin + 1;

			Face_idx.emplace_back(Vector3(p1, p2, p2 + 1));
			Face_idx.emplace_back(Vector3(p1 + 1, p1, p2 + 1));
		}
	}
	
}

void YellowSphere::GenerateDepthAndSilhouette()
{
	Vector3 center = object_params.head(3);
	float radius = mObject_attribute.radius;

	this->generatedDepth.setTo(0);
	this->generatedSilhouette.setTo(0);

	if (center.z() < 200) return;

	int cols = generatedDepth.cols;
	int rows = generatedDepth.rows;

	//����Ƚ����⣬���Բ�����������
	//�ȼ������ĵ�ͶӰ�ĵ㣬Ȼ�����뾶��ͼƬ�еĳ��ȣ�����������������ͼ������ͼ
	Vector2 center_2D = mCamera->world_to_image(center);
	Vector3 TopestPoint = center + radius * Vector3(0, 1, 0);
	Vector2 Topest_2D = mCamera->world_to_image(TopestPoint);

	float R_2D = sqrt((Topest_2D.x() - center_2D.x())*(Topest_2D.x() - center_2D.x()) + (Topest_2D.y() - center_2D.y())*(Topest_2D.y() - center_2D.y()));

	int row_min = (center_2D.y() - R_2D - 1) < 0 ? 0 : (center_2D.y() - R_2D - 1);
	int row_max = (center_2D.y() + R_2D + 1) > (rows - 1) ? (rows - 1) : (center_2D.y() + R_2D + 1);
	int col_min = (center_2D.x() - R_2D - 1) < 0 ? 0 : (center_2D.x() - R_2D - 1);
	int col_max = (center_2D.x() + R_2D + 1) > (cols - 1) ? (cols - 1) : (center_2D.x() + R_2D + 1);

	int count = 0;
	for (int row = row_min; row < row_max; ++row) {
		for (int col = col_min; col < col_max; ++col) {

			float distance = sqrt((row - center_2D.y())*(row - center_2D.y()) + (col - center_2D.x())*(col - center_2D.x()));

			if (distance <= R_2D)
			{
				generatedSilhouette.at<uchar>(row, col) = 255;
				count++;
				//�������ֵ
				float d = center.z() - radius * sqrt(R_2D * R_2D - distance * distance) / R_2D;
				generatedDepth.at<ushort>(row, col) = d;
			}
		}
	}
}

void YellowSphere::ShowDepth()
{
	cv::Mat tmp_depth = generatedDepth.clone();
	tmp_depth.setTo(0, generatedSilhouette == 0);
	double max, min;
	cv::minMaxIdx(tmp_depth, &min, &max);
	cv::Mat normal_map;
	tmp_depth.convertTo(normal_map, CV_8U, 255.0 / (max - min), -min * 255.0 / (max - min));  //�Ҷ�����
	cv::Mat junheng_Map;
	cv::equalizeHist(normal_map, junheng_Map);   //ֱ��ͼ������߶Աȶ�
	cv::Mat color_map;
	cv::applyColorMap(junheng_Map, color_map, cv::COLORMAP_COOL);  //α��ɫ
	color_map.setTo(cv::Scalar(255, 255, 255), generatedSilhouette == 0);  //�ֲ����������ط�����
	cv::flip(color_map, color_map, 0);
	cv::imshow("YellowSphere���ɵ����ͼ", color_map);
}

void YellowSphere::Update(const Eigen::VectorXf& params)
{
	object_params = params;
	this->UpdateVerticesAndNormal();
}

void YellowSphere::UpdateVerticesAndNormal()
{
	float width = mCamera->width();
	float heigh = mCamera->height();

	Visible_2D.clear();
	Visible_3D.points.clear();


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
}



bool YellowSphere::Is_inside(const Eigen::VectorXf& p)
{
	float R = mObject_attribute.radius;
	Vector3 C = Vector3(object_params(0), object_params(1), object_params(2));

	return (C - p).norm() < R ? true : false;
}

Eigen::VectorXf YellowSphere::FindTarget(const Eigen::VectorXf& p)
{
	float R = mObject_attribute.radius;
	Vector3 C = Vector3(object_params(0), object_params(1), object_params(2));

	Vector3 dir = (p - C).normalized();

	return C + R * dir;
}