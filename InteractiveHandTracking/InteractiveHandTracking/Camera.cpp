#include "Camera.h"


Camera::Camera(RuntimeType type) :_type(type)
{
	switch (type)
	{
	case Dataset_MSRA_14:
		//��˵���ļ��еó�
		_width = 320;
		_height = 240;
		_focal_length_x = 241.42f;
		_focal_length_y = 241.42f;
		_CameraCenterX = 160.0f;
		_CameraCenterY = 120.0f;
		break;
	case Dataset_MSRA_15:
		//��˵���ļ��еó�
		_width = 320;
		_height = 240;
		_focal_length_x = 241.42f;
		_focal_length_y = 241.42f;
		_CameraCenterX = 160.0f;
		_CameraCenterY = 120.0f;
		break;
	case Guess_who:
		_width = 320;
		_height = 240;
		_focal_length_x = 224.502f;
		_focal_length_y = 230.494f;
		_CameraCenterX = 160.0f;
		_CameraCenterY = 120.0f;
		break;
	case Handy_teaser:
		//���������hmodel�����е�camera����һ��
		_width = 320;
		_height = 240;
		_focal_length_x = 224.502f;
		_focal_length_y = 230.494f;
		_CameraCenterX = 160.0f;
		_CameraCenterY = 120.0f;
		break;
	case ICVL:
		_width = 320;
		_height = 240;
		_focal_length_x = 240.99f;
		_focal_length_y = 240.96f;
		_CameraCenterX = 160.0f;
		_CameraCenterY = 120.0f;
		break;
	case NYU:
		_width = 640;
		_height = 480;
		_focal_length_x = 588.036865f;
		_focal_length_y = 587.075073f;
		_CameraCenterX = 320.0f;
		_CameraCenterY = 240.0f;
		break;
	default:
		//�������в�������ʹ��MATLAB�Դ�������궨������в���
		_width = 320;
		_height = 240;
		_zFar = 800;
		_zNear = 50;
		_CameraCenterX = 160.0f;    //ͨ��������� _width/2
		_CameraCenterY = 120.0f;    //ͨ��������� _height/2
		_focal_length_x = 224.502f;
		_focal_length_y = 230.494f;
		break;
	}

	///--- Assemble projection matrix
	auto kinectproj = [=]() {
		Matrix3 cam_matrix = Matrix3::Zero();
		cam_matrix(0, 0) = _focal_length_x; /// FocalLength X
		cam_matrix(1, 1) = _focal_length_y; /// FocalLength Y
		cam_matrix(0, 2) = _CameraCenterX;      /// CameraCenter X
		cam_matrix(1, 2) = _CameraCenterY;     /// CameraCenter Y
		cam_matrix(2, 2) = 1.0;
		return cam_matrix;
	};
	proj = kinectproj();
	iproj = proj.inverse();
}


//���������(height() - j - 1)�����ǿ��ǵ����¹�ϵ��(ע��������������ͼ����cv::filp(img,img,0)��x�ᷭת�󣬾Ͳ���Ҫʹ��(height() - j - 1)�ˣ�
//camera����ϵ
//              |y
//              |
//              |________x              ��Kinect������ͷ��������ϵһ���������˳��ӽǣ�
//             /                       ����Ҫע����㣬Kinec���������ͼ�ǣ�   O----------x
//          z /                                                                |                  |y
//                                                                             |y                 |
//                                      ��ˣ�Ҫ�����ͼ����cv::filp(img,img,0)��x�ᷭת���õ�    O-------x����������֮���ٽ���ת��������ϵ
Eigen::Vector3f Camera::depth_to_world(float col, float row, float depth) {
	//Eigen::Vector3f wrld = iproj * Eigen::Vector3f(i*depth, (height() - j - 1)*depth, depth);
	Eigen::Vector3f wrld = iproj * Eigen::Vector3f(col*depth, row*depth, depth);
	return wrld;
}


//����Ƿ�ͶӰ����������
Eigen::Vector3f Camera::unproject(int i, int j, float depth) {
	return pixel_to_image_plane(i, j) * depth;
}

Eigen::Vector3f Camera::pixel_to_image_plane(int i, int j) {
	float x = (i - proj(0, 2)) / proj(0, 0);
	float y = (j - proj(1, 2)) / proj(1, 1);
	return Eigen::Vector3f(x, y, 1);
}

//�����ͶӰ����������
Eigen::Vector2f Camera::world_to_image(const Eigen::Vector3f& wrld) {
	float x = wrld[0] / wrld[2];
	float y = wrld[1] / wrld[2];
	x = x*proj(0, 0) + proj(0, 2);
	y = y*proj(1, 1) + proj(1, 2);
	return Eigen::Vector2f(x, y);
}

Matrix4 Camera::view_projection_matrix() {
	///--- Intrinsics matrix
	Matrix3& K = proj;
	int w = this->width();
	int h = this->height();

	Matrix4 mat = Matrix4::Identity();
	mat(0, 0) = 2.0f / (float)w*K(0, 0); // use camera instrinsics and convert to GL [0,h] => [-1,1]
	mat(0, 2) = (2.0f / (float)w*(K(0, 2) + 0.5f)) - 1.0f; // 0.5 offset as GL pixel middle point is at 0.5,0.5
														   // Y
	mat(1, 1) = 2.0f / (float)h*K(1, 1); // use camera instrinsics and convert to GL [0,h] => [-1,1]
	mat(1, 2) = (2.0f / (float)h*(K(1, 2) + 0.5f)) - 1.0f;
	// Z
	mat(2, 2) = (_zFar + _zNear) / (_zFar - _zNear);
	mat(2, 3) = -2.0f*_zFar*_zNear / (_zFar - _zNear);
	// W
	mat(3, 2) = 1; // not as in GL where it would be -1
	mat(3, 3) = 0;

	return mat;
}

Matrix_2x3 Camera::projection_jacobian(const Eigen::Vector3f &p) {
	Matrix_2x3 M;
	M << _focal_length_x / p.z(), 0, -p.x() * _focal_length_x / (p.z()*p.z()),
		0, _focal_length_y / p.z(), -p.y() * _focal_length_y / (p.z()*p.z());
	return M;
}

