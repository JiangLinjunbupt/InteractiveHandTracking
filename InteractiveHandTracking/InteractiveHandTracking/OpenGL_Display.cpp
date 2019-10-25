#include"OpenGL_Display.h"


namespace DS
{
	InputManager* mInputManager;

	bool show_handmodel = true;
	bool pause = false;
	bool track = true;
	int idx = 0;
	Control control;

	//定义光照
	void light() {

		GLfloat light_position[] = { 1.0f,1.0f,1.0f,0.0f };//1.0表示光源为点坐标x,y,z
		GLfloat white_light[] = { 1.0f,1.0f,1.0f,1.0f };   //光源的颜色
		GLfloat lmodel_ambient[] = { 0.2f,0.2f,0.2f,1.0f };//微弱环境光，使物体可见
		glShadeModel(GL_SMOOTH);//GL_SMOOTH

		glLightfv(GL_LIGHT0, GL_POSITION, light_position);//光源编号-7，光源特性，参数数据
		glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
		glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient); //指定全局的环境光，物体才能可见//*/

		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_DEPTH_TEST);

		//glEnable(GL_LIGHTING);
		//glEnable(GL_NORMALIZE);
		//// 定义太阳光源，它是一种白色的光源  
		//GLfloat sun_light_position[] = { 0.0f, 0.0f, 0.0f, 1.0f };
		//GLfloat sun_light_ambient[] = { 0.25f, 0.25f, 0.15f, 1.0f };
		//GLfloat sun_light_diffuse[] = { 0.7f, 0.7f, 0.55f, 1.0f };
		//GLfloat sun_light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

		//glLightfv(GL_LIGHT0, GL_POSITION, sun_light_position); //指定第0号光源的位置   
		//glLightfv(GL_LIGHT0, GL_AMBIENT, sun_light_ambient); //GL_AMBIENT表示各种光线照射到该材质上，  
		//													 //经过很多次反射后最终遗留在环境中的光线强度（颜色）  
		//glLightfv(GL_LIGHT0, GL_DIFFUSE, sun_light_diffuse); //漫反射后~~  
		//glLightfv(GL_LIGHT0, GL_SPECULAR, sun_light_specular);//镜面反射后~~~  

		//glEnable(GL_LIGHT0); //使用第0号光照   
	}

	//定义窗口大小重新调整
	void reshape(int width, int height) {

		GLfloat fieldOfView = 90.0f;
		glViewport(0, 0, (GLsizei)width, (GLsizei)height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(fieldOfView, (GLfloat)width / (GLfloat)height, 0.1, 500.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	//键盘鼠标响应函数
	void keyboardDown(unsigned char key, int x, int y) {
		if (key == 'q') exit(0);
		if (key == 'h') show_handmodel = !show_handmodel;
		//if (key == 'd') show_Dataset = !show_Dataset;
		if (key == 'p') pause = !pause;
		if (key == 't') track = !track;
	}
	void mouseClick(int button, int state, int x, int y) {
		control.mouse_click = 1;
		control.x = x;
		control.y = y;
	}
	void mouseMotion(int x, int y) {
		control.rotx = (x - control.x)*0.05f;
		control.roty = (y - control.y)*0.05f;

		if (control.roty > 1.57) control.roty = 1.57;
		if (control.roty < -1.57) control.roty = -1.57;
		//cout << control.rotx << " " << control.roty << endl;
		glutPostRedisplay();
	}

#pragma region SetOfDraw
	//一系列绘制函数
	void draw_HandPointCloud()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		if (mInputManager->mImage_InputData.hand.pointcloud.points.size() > 0)
		{
			glColor3f(0.0f, 1.0f, 0.0f);
			for (int i = 0; i < mInputManager->mImage_InputData.hand.pointcloud.points.size(); i++) {
				glPushMatrix();
				glTranslatef(mInputManager->mImage_InputData.hand.pointcloud.points[i].x,
					mInputManager->mImage_InputData.hand.pointcloud.points[i].y,
					mInputManager->mImage_InputData.hand.pointcloud.points[i].z);
				glutSolidSphere(2, 10, 10);
				glPopMatrix();
			}
		}
	}
	void draw_ObjectCloud()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		if (mInputManager->mImage_InputData.item.pointcloud.points.size() > 0)
		{
			glColor3f(1.0f, 1.0f, 0.0f);
			for (int i = 0; i < mInputManager->mImage_InputData.item.pointcloud.points.size(); i++) {
				glPushMatrix();
				glTranslatef(mInputManager->mImage_InputData.item.pointcloud.points[i].x,
					mInputManager->mImage_InputData.item.pointcloud.points[i].y,
					mInputManager->mImage_InputData.item.pointcloud.points[i].z);
				glutSolidSphere(2, 10, 10);
				glPopMatrix();
			}
		}
	}
	void draw_Sphere()
	{
		glEnable(GL_LIGHT0);
		glEnable(GL_LIGHTING);
		GLfloat Sphere_ambient[] = { 1,1,0,1 };
		GLfloat Sphere_specular[] = { 0.5, 0.5, 0.5, 1.0 };
		GLfloat Sphere_shin[] = { 10 };
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, Sphere_ambient);
		glMaterialfv(GL_FRONT, GL_SPECULAR, Sphere_specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, Sphere_shin);

		glPushMatrix();
		glTranslatef(mInputManager->mImage_InputData.item.center(0),
			mInputManager->mImage_InputData.item.center(1),
			mInputManager->mImage_InputData.item.center(2));
		glutSolidSphere(25, 20, 20);
		glPopMatrix();
	}
	void draw_HandPointCloudNormal()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		if (mInputManager->mImage_InputData.hand.pointcloud.points.size() > 0)
		{
			int scale = 20;
			glColor3f(0.0f, 0.0f, 1.0f);
			glLineWidth(2);
			glBegin(GL_LINES);
			for (int i = 0; i < mInputManager->mImage_InputData.hand.pointcloud.points.size(); i++) {
				glVertex3f(mInputManager->mImage_InputData.hand.pointcloud.points[i].x,
					mInputManager->mImage_InputData.hand.pointcloud.points[i].y,
					mInputManager->mImage_InputData.hand.pointcloud.points[i].z);
				glVertex3f(mInputManager->mImage_InputData.hand.pointcloud.points[i].x + scale * mInputManager->mImage_InputData.hand.pointcloud.points[i].normal_x,
					mInputManager->mImage_InputData.hand.pointcloud.points[i].y + scale * mInputManager->mImage_InputData.hand.pointcloud.points[i].normal_y,
					mInputManager->mImage_InputData.hand.pointcloud.points[i].z + scale * mInputManager->mImage_InputData.hand.pointcloud.points[i].normal_z);
			}
			glEnd();
		}
	}
	void draw_ObjectCloudNormal()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		if (mInputManager->mImage_InputData.item.pointcloud.points.size() > 0)
		{
			int scale = 20;
			glColor3f(0.0f, 0.0f, 1.0f);
			glLineWidth(2);
			glBegin(GL_LINES);
			for (int i = 0; i < mInputManager->mImage_InputData.item.pointcloud.points.size(); i++) {
				glVertex3f(mInputManager->mImage_InputData.item.pointcloud.points[i].x,
					mInputManager->mImage_InputData.item.pointcloud.points[i].y,
					mInputManager->mImage_InputData.item.pointcloud.points[i].z);
				glVertex3f(mInputManager->mImage_InputData.item.pointcloud.points[i].x + scale * mInputManager->mImage_InputData.item.pointcloud.points[i].normal_x,
					mInputManager->mImage_InputData.item.pointcloud.points[i].y + scale * mInputManager->mImage_InputData.item.pointcloud.points[i].normal_y,
					mInputManager->mImage_InputData.item.pointcloud.points[i].z + scale * mInputManager->mImage_InputData.item.pointcloud.points[i].normal_z);
			}
			glEnd();
		}
	}
	void draw_Coordinate()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		////x
		//glLineWidth(5);
		//glColor3f(1.0, 0.0, 0.0);
		//glBegin(GL_LINES);
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1), _dataframe->palm_Center(2));
		//glVertex3f(_dataframe->palm_Center(0) + 100, _dataframe->palm_Center(1), _dataframe->palm_Center(2));
		//glEnd();

		////y
		//glLineWidth(5);
		//glColor3f(0.0, 1.0, 0.0);
		//glBegin(GL_LINES);
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1), _dataframe->palm_Center(2));
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1) + 100, _dataframe->palm_Center(2));
		//glEnd();

		////z
		//glLineWidth(5);
		//glColor3f(0.0, 0.0, 1.0);
		//glBegin(GL_LINES);
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1), _dataframe->palm_Center(2));
		//glVertex3f(_dataframe->palm_Center(0), _dataframe->palm_Center(1), _dataframe->palm_Center(2) + 100);
		//glEnd();

		//x
		glLineWidth(5);
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
		glVertex3f(90, -50, 450);
		glVertex3f(90 + 100, -50, 450);
		glEnd();

		//y
		glLineWidth(5);
		glColor3f(0.0, 1.0, 0.0);
		glBegin(GL_LINES);
		glVertex3f(90, -50, 450);
		glVertex3f(90, -50 + 100, 450);
		glEnd();

		//z
		glLineWidth(5);
		glColor3f(0.0, 0.0, 1.0);
		glBegin(GL_LINES);
		glVertex3f(90, -50, 450);
		glVertex3f(90, -50, 450 + 100);
		glEnd();
	}
#pragma endregion SetOfDraw
	void draw() {

		//glClearColor(0.5, 0.5, 0.5, 1);
		glClearColor(1, 1, 1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glMatrixMode(GL_MODELVIEW);
		gluPerspective(180, 1.5, -1000, 1000);
		glLoadIdentity();
		control.gx = mInputManager->mImage_InputData.hand.center(0);
		control.gy = mInputManager->mImage_InputData.hand.center(1);
		control.gz = mInputManager->mImage_InputData.hand.center(2);

		//这个值是根据palm_Center设置的，因为如果使用palm_Center的话，跳动会变得非常明显
		if (mInputManager->getRuntimeType() == REALTIME)
		{
			control.gx = 90;
			control.gy = -30;
			control.gz = 450;
		}

		double r = 250;
		double x = r*cos(control.roty)*sin(control.rotx);
		double y = r*sin(control.roty);
		double z = r*cos(control.roty)*cos(control.rotx);
		//cout<< x <<" "<< y <<" " << z<<endl;
		gluLookAt(x + control.gx, y + control.gy, z + control.gz, control.gx, control.gy, control.gz, 0.0, 1.0, 0.0);//个人理解最开始是看向-z的，之后的角度是在global中心上叠加的，所以要加

		//draw_HandPointCloud();
		//draw_HandPointCloudNormal();
		//draw_ObjectCloud();
		//draw_ObjectCloudNormal();
		//draw_Coordinate();
		//draw_Sphere();

		glFlush();
		glutSwapBuffers();
	}

	void idle() {

		if (mInputManager->fetchInputData())
		{
			mInputManager->ShowImage_input(true, true, true);
		}

		glutPostRedisplay();
	}


	//GL初始化函数
	void InitializeGlutCallbacks()
	{
		glutKeyboardFunc(keyboardDown);
		glutMouseFunc(mouseClick);
		glutMotionFunc(mouseMotion);
		glutReshapeFunc(reshape);
		glutDisplayFunc(draw);
		glutIdleFunc(idle);
		glutIgnoreKeyRepeat(true); // ignore keys held down
	}
	void initScene(int width, int height) {
		reshape(width, height);

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClearDepth(1.0f);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LEQUAL);
		light();
	}
	void init(int argc, char* argv[]) {
		// 初始化GLUT
		glutInit(&argc, argv);

		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
		glutInitWindowSize(800, 600);
		glutInitWindowPosition(100, 100);
		glutCreateWindow("SHOW RESULT");
		InitializeGlutCallbacks();
		initScene(800, 600);
	}

	void start() {
		// 通知开始GLUT的内部循环
		glutMainLoop();
	}
}