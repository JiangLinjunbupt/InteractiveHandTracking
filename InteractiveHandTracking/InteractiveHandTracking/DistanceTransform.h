#pragma once
#include "opencv2/core/core.hpp"    
#include<iostream>
/*@article{felzenszwalb_12,
Author = { Felzenszwalb, Pedro F and Huttenlocher, Daniel P },
Journal = { Theory of Computing },
Title = { Distance Transforms of Sampled Functions },
Year = { 2012 }}
github ： https://github.com/ataiya/dtform
效果应该是和opencv的DistanceTransform函数相同，但是这个代码不仅可以找到最佳距离，还可以得到对应的点，而且时间复杂度都是线性*/



//测试代码（放在源.cpp中）
//====>>
//distanceTransform对象创建，初始化。
//cv::Mat  hand = handfinder.sensor_hand_silhouette;
//cv::Mat hand2;
//distance_transform.exec(hand.data, 125);
////hand2 = distance_transform.dsts_image();
////normalize(hand2, hand2, 0, 1, NORM_MINMAX);  //如果要看距离变换后的图像，这里需要使用normalize归一化
////imshow(" hand2", hand2);         
//
//circle(hand, cvPoint(100, 200), 10, Scalar(255, 255, 255), -1, 8);
//circle(hand, cvPoint(400, 200), 10, Scalar(255, 255, 255), -1, 8);
//
//int idx = distance_transform.idx_at(200, 100);       //这里注意  distance_transform.idx_at 查询的是，第几行，第几列。而cvPoint(100, 200) 的坐标是x和y ，分别表示，第几列，第几行。注意这个差别。
//int idx2 = distance_transform.idx_at(200, 400);
//
//line(hand, cvPoint(100, 200), cvPoint(idx % 512, idx / 512), 0, 5);
//line(hand, cvPoint(400, 200), cvPoint(idx2 % 512, idx2 / 512), 0, 5);
//
//circle(hand, cvPoint(idx % 512, idx / 512), 10, Scalar(255, 255, 255), -1, 8);
//circle(hand, cvPoint(idx2 % 512, idx2 / 512), 10, Scalar(255, 255, 255), -1, 8);
//imshow("hand", hand);
//cvWaitKey(1);


class DistanceTransform {
private:
	int width;
	int height;
	float* v = NULL;
	float* z = NULL;
	float* DTTps = NULL;
	int* ADTTps = NULL;
	float* realDT = NULL;
	int* realADT = NULL; ///< stores closest point ids (float just to simplify CUDA)

public:
	void init(int width, int height) {
		this->width = width;
		this->height = height;
		v = new float[width*height];
		z = new float[width*height];
		DTTps = new float[width*height];
		ADTTps = new int[width*height];
		realADT = new int[width*height];
		realDT = new float[width*height];
	}
	void cleanup() {
		delete[] v;
		delete[] z;
		delete[] DTTps;
		delete[] ADTTps;
		delete[] realADT;
		delete[] realDT;
	}


	float* dsts_image_ptr() { return realDT; }
	int* idxs_image_ptr() { return realADT; }
	cv::Mat dsts_image() { return cv::Mat(height, width, CV_32FC1 /*float*/, realDT); }
	cv::Mat idxs_image() { return cv::Mat(height, width, CV_32SC1 /*int*/, realADT); }
	int dst_at(int row, int col) { return realDT[row*width + col]; }
	int idx_at(int row, int col) { return realADT[row*width + col]; }


public:
	/// @pars row major uchar binary image. White pixels are "data" and 
	/// label_image[i] > mask_th decides what data is.
	void exec(unsigned char* label_image, int mask_th = 125)
	{
		//#pragma omp parallel
		{
			//#pragma omp for
			for (int i = 0; i < width*height; ++i)
			{
				if (label_image[i] < mask_th)
				{
					realDT[i] = FLT_MAX;
				}
				else
				{
					realDT[i] = 0.0f;
				}
			}

			/////////////////////////////////////////////////////////////////
			/// DT and ADT
			/////////////////////////////////////////////////////////////////

			//First PASS (rows)
			//#pragma omp for
			for (int row = 0; row<height; ++row)
			{
				unsigned int k = 0;
				unsigned int indexpt1 = row*width;
				v[indexpt1] = 0;
				z[indexpt1] = FLT_MIN;
				z[indexpt1 + 1] = FLT_MAX;
				for (int q = 1; q<width; ++q)
				{
					float sp1 = float(realDT[(indexpt1 + q)] + (q*q));
					unsigned int index2 = indexpt1 + k;
					unsigned int vk = v[index2];
					float s = (sp1 - float(realDT[(indexpt1 + vk)] + (vk*vk))) / float((q - vk) << 1);
					while (s <= z[index2] && k > 0)
					{
						k--;
						index2 = indexpt1 + k;
						vk = v[index2];
						s = (sp1 - float(realDT[(indexpt1 + vk)] + (vk*vk))) / float((q - vk) << 1);
					}
					k++;
					index2 = indexpt1 + k;
					v[index2] = q;
					z[index2] = s;
					z[index2 + 1] = FLT_MAX;
				}
				k = 0;
				for (int q = 0; q<width; ++q)
				{
					while (z[indexpt1 + k + 1]<q)
						k++;
					unsigned int index2 = indexpt1 + k;
					unsigned int vk = v[index2];
					float tp1 = float(q) - float(vk);
					DTTps[indexpt1 + q] = tp1*tp1 + float(realDT[(indexpt1 + vk)]);
					ADTTps[indexpt1 + q] = indexpt1 + vk;
				}
			}

			//--- Second PASS (columns)
			//#pragma omp for
			for (int col = 0; col<width; ++col)
			{
				unsigned int k = 0;
				unsigned int indexpt1 = col*height;
				v[indexpt1] = 0;
				z[indexpt1] = FLT_MIN;
				z[indexpt1 + 1] = FLT_MAX;
				for (int row = 1; row<height; ++row)
				{
					float sp1 = float(DTTps[col + row*width] + (row*row));
					unsigned int index2 = indexpt1 + k;
					unsigned int vk = v[index2];
					float s = (sp1 - float(DTTps[col + vk*width] + (vk*vk))) / float((row - vk) << 1);
					while (s <= z[index2] && k > 0)
					{
						k--;
						index2 = indexpt1 + k;
						vk = v[index2];
						s = (sp1 - float(DTTps[col + vk*width] + (vk*vk))) / float((row - vk) << 1);
					}
					k++;
					index2 = indexpt1 + k;
					v[index2] = row;
					z[index2] = s;
					z[index2 + 1] = FLT_MAX;
				}
				k = 0;
				for (int row = 0; row<height; ++row)
				{
					while (z[indexpt1 + k + 1]<row)
						k++;
					unsigned int index2 = indexpt1 + k;
					unsigned int vk = v[index2];

					///// Also compute the distance value
					float tp1 = float(row) - float(vk);
					realDT[col + row*width] = sqrtf(tp1*tp1 + DTTps[col + vk*width]);

					realADT[col + row*width] = ADTTps[col + vk*width];
				}
			}
		} ///< OPENMP
	}
};


