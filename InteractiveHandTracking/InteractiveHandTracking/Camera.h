#pragma once
#include"Types.h"

class Camera
{
private:
	Matrix3 proj;
	Matrix3 iproj;

	int      _fps = 60; ///< frame per second
	int      _width;
	int      _height;
	float    _zNear; ///< near clip plane (mm)
	float    _zFar;  ///< far clip plane (mm)
	float    _CameraCenterX;
	float    _CameraCenterY;
	float    _focal_length_x = nan();
	float    _focal_length_y = nan();

public:
	Camera(RuntimeType type);
	RuntimeType _type;
public:
	int width() const { return _width; }
	int height() const { return _height; }
	float focal_length_x() const { return _focal_length_x; }
	float focal_length_y() const { return _focal_length_y; }
	float cameraCenterX() const { return _CameraCenterX; }
	float cameraCenterY() const { return _CameraCenterY; }
	float zSpan() const { return (_zFar - _zNear); }
	float zNear() const { return _zNear; }
	float zFar() const { return _zFar; }
	bool is_valid(float depth) { return ((depth>_zNear) && (depth<_zFar)); }
	int FPS() const { return _fps; }
	const Matrix3& inv_projection_matrix() const { return iproj; }

public:
	/// View+Projection matrix (part of MVP) that considers the sensor intrinsics
	Matrix4 view_matrix() { return Matrix4::Identity(); }
	Matrix4 view_projection_matrix();
	Matrix_2x3 projection_jacobian(const Eigen::Vector3f& p);

public:
	Eigen::Vector3f depth_to_world(float col, float row, float depth);

	////这个是投影的正常步骤，但是投影到图像平面后，还需要经过关于x轴的反转，原因见cpp文件
	Eigen::Vector2f world_to_image(const Eigen::Vector3f& wrld);

	///这个是反投影的正常步骤，但是用的时候需要用 depth_to_world ，原因见cpp文件
	Eigen::Vector3f unproject(int i, int j, float depth);
	Eigen::Vector3f pixel_to_image_plane(int i, int j);
};
