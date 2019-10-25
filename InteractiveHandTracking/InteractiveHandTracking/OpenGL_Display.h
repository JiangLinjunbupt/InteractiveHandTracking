#pragma once
#include <GL\freeglut.h>
#include <time.h>
#include<queue>
#include<vector>
#include<chrono>

#include"InputManager.h"

using namespace std::chrono;
namespace DS {
	extern InputManager* mInputManager;

	extern bool show_handmodel;
	extern bool pause;
	extern bool track;
	extern Control control;
	extern int idx;
	//定义光照
	void light();

	//定义窗口大小重新调整
	void reshape(int width, int height);

	//键盘鼠标响应函数
	void keyboardDown(unsigned char key, int x, int y);
	void mouseClick(int button, int state, int x, int y);
	void mouseMotion(int x, int y);


	//一系列绘制函数
	void draw_HandPointCloud();
	void draw_HandPointCloudNormal();
	void draw_ObjectCloud();
	void draw_ObjectCloudNormal();
	void draw_Coordinate();
	void draw_Sphere();
	void draw();

	//OpenGL控制函数
	void idle();


	//GL初始化函数
	void InitializeGlutCallbacks();
	void initScene(int width, int height);
	void init(int argc, char* argv[]);

	void start();
}