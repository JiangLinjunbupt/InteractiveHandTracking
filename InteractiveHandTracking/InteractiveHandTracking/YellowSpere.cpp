#include"Interacted_Object.h"


YellowSphere::YellowSphere(Camera* camera) : Interacted_Object(camera)
{
	this->radius = 25;
	this->center = Vector3(90.0f, -30.0f, 450.0f);
	this->color = Vector3(1.0f, 1.0f, 0.0f);
}

void YellowSphere::GenerateDepthAndSilhouette()
{
	this->generatedDepth.setTo(0);
	this->generatedSilhouette.setTo(0);

	if (center.z() < 200) return;

	int cols = generatedDepth.cols;
	int rows = generatedDepth.rows;

	//球体比较特殊，可以不用三角网格
	//先计算中心点投影的点，然后计算半径到图片中的长度，最后根据这个计算深度图和轮廓图
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
				//生成深度值
				float d = center.z() - radius * sqrt(R_2D * R_2D - distance * distance) / R_2D;
				generatedDepth.at<ushort>(row, col) = d;
			}
		}
	}
}

Vector3 YellowSphere::FindCorrespondintPoint(const Vector3& p)
{
	Vector3 centerTopoint = p - center;
	centerTopoint.normalize();

	return center + radius * centerTopoint;
}

void YellowSphere::ShowDepth()
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
	cv::imshow("YellowSphere生成的深度图", color_map);
}