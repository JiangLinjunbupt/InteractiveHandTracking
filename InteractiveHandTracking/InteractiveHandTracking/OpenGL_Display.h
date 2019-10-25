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
	//�������
	void light();

	//���崰�ڴ�С���µ���
	void reshape(int width, int height);

	//���������Ӧ����
	void keyboardDown(unsigned char key, int x, int y);
	void mouseClick(int button, int state, int x, int y);
	void mouseMotion(int x, int y);


	//һϵ�л��ƺ���
	void draw_HandPointCloud();
	void draw_HandPointCloudNormal();
	void draw_ObjectCloud();
	void draw_ObjectCloudNormal();
	void draw_Coordinate();
	void draw_Sphere();
	void draw();

	//OpenGL���ƺ���
	void idle();


	//GL��ʼ������
	void InitializeGlutCallbacks();
	void initScene(int width, int height);
	void init(int argc, char* argv[]);

	void start();
}