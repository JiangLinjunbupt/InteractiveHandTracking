#pragma once
#include"Types.h"
#include"Camera.h"

class Interacted_Object
{
protected:
	Camera* mCamera;
public:
	cv::Mat generatedDepth;
	cv::Mat generatedSilhouette;

	Interacted_Object(Camera* camera) :mCamera(camera) {
		generatedDepth = cv::Mat(cv::Size(mCamera->width(), mCamera->height()), CV_16UC1, cv::Scalar(0));
		generatedSilhouette = cv::Mat(cv::Size(mCamera->width(), mCamera->height()), CV_8UC1, cv::Scalar(0));
	};
	virtual ~Interacted_Object(){}
	virtual void GenerateDepthAndSilhouette() = 0;
	virtual Vector3 FindCorrespondintPoint(const Vector3& p) = 0;
	virtual void ShowDepth() = 0;
};


class YellowSphere : public Interacted_Object
{
public:
	float radius;
	Vector3 center;
	Vector3 color;

public:
	YellowSphere(Camera* camera);
	virtual ~YellowSphere() {};
	
	void GenerateDepthAndSilhouette();
	Vector3 FindCorrespondintPoint(const Vector3& p);
	void ShowDepth();
};