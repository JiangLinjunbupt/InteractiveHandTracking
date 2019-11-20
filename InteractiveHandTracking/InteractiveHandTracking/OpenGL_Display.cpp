#include"OpenGL_Display.h"


namespace DS
{
	TrackingManager* mTrackingManager;

	bool show_handmodel = true;
	bool pause = false;
	bool track = false;
	int idx = 0;
	Control control;

	//�������
	void light() {

		GLfloat light_position[] = { 1.0f,1.0f,1.0f,0.0f };//1.0��ʾ��ԴΪ������x,y,z
		GLfloat white_light[] = { 1.0f,1.0f,1.0f,1.0f };   //��Դ����ɫ
		GLfloat lmodel_ambient[] = { 0.2f,0.2f,0.2f,1.0f };//΢�������⣬ʹ����ɼ�
		glShadeModel(GL_SMOOTH);//GL_SMOOTH

		glLightfv(GL_LIGHT0, GL_POSITION, light_position);//��Դ���-7����Դ���ԣ���������
		glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
		glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
		glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient); //ָ��ȫ�ֵĻ����⣬������ܿɼ�//*/

		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_DEPTH_TEST);

		//glEnable(GL_LIGHTING);
		//glEnable(GL_NORMALIZE);
		//// ����̫����Դ������һ�ְ�ɫ�Ĺ�Դ  
		//GLfloat sun_light_position[] = { 0.0f, 0.0f, 0.0f, 1.0f };
		//GLfloat sun_light_ambient[] = { 0.25f, 0.25f, 0.15f, 1.0f };
		//GLfloat sun_light_diffuse[] = { 0.7f, 0.7f, 0.55f, 1.0f };
		//GLfloat sun_light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

		//glLightfv(GL_LIGHT0, GL_POSITION, sun_light_position); //ָ����0�Ź�Դ��λ��   
		//glLightfv(GL_LIGHT0, GL_AMBIENT, sun_light_ambient); //GL_AMBIENT��ʾ���ֹ������䵽�ò����ϣ�  
		//													 //�����ܶ�η�������������ڻ����еĹ���ǿ�ȣ���ɫ��  
		//glLightfv(GL_LIGHT0, GL_DIFFUSE, sun_light_diffuse); //�������~~  
		//glLightfv(GL_LIGHT0, GL_SPECULAR, sun_light_specular);//���淴���~~~  

		//glEnable(GL_LIGHT0); //ʹ�õ�0�Ź���   
	}

	//���崰�ڴ�С���µ���
	void reshape(int width, int height) {

		GLfloat fieldOfView = 90.0f;
		glViewport(0, 0, (GLsizei)width, (GLsizei)height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(fieldOfView, (GLfloat)width / (GLfloat)height, 0.1, 500.0);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
	}

	//���������Ӧ����
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
	//һϵ�л��ƺ���
	void draw_HandPointCloud()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		int size = mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points.size();
		if (size > 0)
		{
			glColor3f(0.0f, 1.0f, 0.0f);
			for (int i = 0; i < size; i++) {
				glPushMatrix();
				glTranslatef(mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].x,
					mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].y,
					mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].z);
				glutSolidSphere(2, 10, 10);
				glPopMatrix();
			}
		}
	}
	void draw_ObjectCloud()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		for (size_t obj_idx = 0; obj_idx < mTrackingManager->mInputManager->mInputData.image_data.item.size(); ++obj_idx)
		{
			int size = mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points.size();
			if (size > 0)
			{
				glPointSize(5);
				glColor3f(mTrackingManager->mInteracted_Object[obj_idx]->mObject_attribute.color(0),
					mTrackingManager->mInteracted_Object[obj_idx]->mObject_attribute.color(1),
					mTrackingManager->mInteracted_Object[obj_idx]->mObject_attribute.color(2));
				glBegin(GL_POINTS);
				for (int i = 0; i < size; i++) {
					glVertex3f(mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].x,
						mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].y,
						mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].z);
				}
				glEnd();
			}
		}
	}
	void draw_HandPointCloudNormal()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		int size = mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points.size();
		if (size > 0)
		{
			int scale = 20;
			glColor3f(0.0f, 0.0f, 1.0f);
			glLineWidth(2);
			glBegin(GL_LINES);
			for (int i = 0; i < size; i++) {
				glVertex3f(mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].x,
					mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].y,
					mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].z);
				glVertex3f(mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].x + scale * mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].normal_x,
					mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].y + scale * mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].normal_y,
					mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].z + scale * mTrackingManager->mInputManager->mInputData.image_data.hand.pointcloud.points[i].normal_z);
			}
			glEnd();
		}
	}
	void draw_ObjectCloudNormal()
	{
		glDisable(GL_LIGHT0);
		glDisable(GL_LIGHTING);
		for (size_t obj_idx = 0; obj_idx < mTrackingManager->mInputManager->mInputData.image_data.item.size(); ++obj_idx)
		{
			int size = mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points.size();
			if (size > 0)
			{
				int scale = 20;
				glColor3f(0.0f, 0.0f, 1.0f);
				glLineWidth(2);
				glBegin(GL_LINES);
				for (int i = 0; i < size; i++) {
					glVertex3f(mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].x,
						mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].y,
						mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].z);
					glVertex3f(mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].x + scale * mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].normal_x,
						mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].y + scale * mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].normal_y,
						mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].z + scale * mTrackingManager->mInputManager->mInputData.image_data.item[obj_idx].pointcloud.points[i].normal_z);
				}
				glEnd();
			}
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

	void draw_Interacted_Object()
	{
		glEnable(GL_LIGHT0);
		glEnable(GL_LIGHTING);
		for (size_t obj_idx = 0; obj_idx < mTrackingManager->mInteracted_Object.size(); ++obj_idx)
		{
			GLfloat Sphere_ambient[] = { mTrackingManager->mInteracted_Object[obj_idx]->mObject_attribute.color(0),
				mTrackingManager->mInteracted_Object[obj_idx]->mObject_attribute.color(1),
				mTrackingManager->mInteracted_Object[obj_idx]->mObject_attribute.color(2),1 };
			GLfloat Sphere_specular[] = { 0.5, 0.5, 0.5, 1.0 };
			GLfloat Sphere_shin[] = { 10 };
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, Sphere_ambient);
			glMaterialfv(GL_FRONT, GL_SPECULAR, Sphere_specular);
			glMaterialfv(GL_FRONT, GL_SHININESS, Sphere_shin);

			if (!mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices.empty())
			{
				int face_size = mTrackingManager->mInteracted_Object[obj_idx]->Face_idx.size();
				glBegin(GL_TRIANGLES);
				for (size_t i = 0; i < face_size; ++i)
				{
					int idx_1 = mTrackingManager->mInteracted_Object[obj_idx]->Face_idx[i](0);
					int idx_2 = mTrackingManager->mInteracted_Object[obj_idx]->Face_idx[i](1);
					int idx_3 = mTrackingManager->mInteracted_Object[obj_idx]->Face_idx[i](2);

					glNormal3f(mTrackingManager->mInteracted_Object[obj_idx]->Final_Normal[idx_1](0),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Normal[idx_1](1),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Normal[idx_1](2));
					glVertex3f(mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices[idx_1](0),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices[idx_1](1),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices[idx_1](2));

					glNormal3f(mTrackingManager->mInteracted_Object[obj_idx]->Final_Normal[idx_2](0),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Normal[idx_2](1),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Normal[idx_2](2));
					glVertex3f(mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices[idx_2](0),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices[idx_2](1),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices[idx_2](2));

					glNormal3f(mTrackingManager->mInteracted_Object[obj_idx]->Final_Normal[idx_3](0),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Normal[idx_3](1),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Normal[idx_3](2));
					glVertex3f(mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices[idx_3](0),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices[idx_3](1),
						mTrackingManager->mInteracted_Object[obj_idx]->Final_Vertices[idx_3](2));
				}
				glEnd();
			}
		}
	}
	void draw_HandModel()
	{
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		{
			glPushMatrix();

			GLfloat mat_ambient[] = { 0.05, 0.05, 0.0, 1.0 };
			GLfloat mat_diffuse[] = { 0.5, 0.4,0.4, 1.0 };
			GLfloat mat_specular[] = { 0.7, 0.04, 0.04, 1.0 };
			GLfloat no_shininess[] = { 0.78125 };

			glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
			glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
			glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
			glMaterialfv(GL_FRONT, GL_SHININESS, no_shininess);

			glBegin(GL_TRIANGLES);
			for (int i = 0; i < mTrackingManager->mHandModel->Face_num; ++i)
			{

				glNormal3f(mTrackingManager->mHandModel->F_normal(i, 0), mTrackingManager->mHandModel->F_normal(i, 1), mTrackingManager->mHandModel->F_normal(i, 2));

				glVertex3f(mTrackingManager->mHandModel->V_Final(mTrackingManager->mHandModel->F(i, 0), 0),
					mTrackingManager->mHandModel->V_Final(mTrackingManager->mHandModel->F(i, 0), 1),
					mTrackingManager->mHandModel->V_Final(mTrackingManager->mHandModel->F(i, 0), 2));

				glVertex3f(mTrackingManager->mHandModel->V_Final(mTrackingManager->mHandModel->F(i, 1), 0),
					mTrackingManager->mHandModel->V_Final(mTrackingManager->mHandModel->F(i, 1), 1),
					mTrackingManager->mHandModel->V_Final(mTrackingManager->mHandModel->F(i, 1), 2));

				glVertex3f(mTrackingManager->mHandModel->V_Final(mTrackingManager->mHandModel->F(i, 2), 0),
					mTrackingManager->mHandModel->V_Final(mTrackingManager->mHandModel->F(i, 2), 1),
					mTrackingManager->mHandModel->V_Final(mTrackingManager->mHandModel->F(i, 2), 2));
			}
			glEnd();
			glPopMatrix();            //��������
		}
	}

	void draw_ContactPoint()
	{
		vector<pair<int, Vector3>> hand_obj_contatct;

		for (int v_id = 0; v_id < mTrackingManager->mHandModel->Vertex_num; ++v_id)
		{
			if (mTrackingManager->mHandModel->contactPoints[v_id] == 1)
			{
				for (int obj_idx = 0; obj_idx < mTrackingManager->mInteracted_Object.size(); ++obj_idx)
				{
					Eigen::Vector3f p(mTrackingManager->mHandModel->V_Final(v_id, 0), 
						mTrackingManager->mHandModel->V_Final(v_id, 1), 
						mTrackingManager->mHandModel->V_Final(v_id, 2));
					if (!mTrackingManager->mInteracted_Object[obj_idx]->Is_inside(p))
					{
						Eigen::Vector3f cor = mTrackingManager->mInteracted_Object[obj_idx]->FindTouchPoint(p);
						hand_obj_contatct.emplace_back(make_pair(v_id, cor));
					}
				}
			}
		}

		int size = hand_obj_contatct.size();

		if (size > 0)
		{
			glBegin(GL_LINES);

			for (int i = 0; i < size; ++i)
			{
				glVertex3f(mTrackingManager->mHandModel->V_Final(hand_obj_contatct[i].first, 0),
					mTrackingManager->mHandModel->V_Final(hand_obj_contatct[i].first, 1),
					mTrackingManager->mHandModel->V_Final(hand_obj_contatct[i].first, 2));

				glVertex3f(hand_obj_contatct[i].second(0), hand_obj_contatct[i].second(1), hand_obj_contatct[i].second(2));
			}

			glEnd();
		}
	}
#pragma endregion SetOfDraw
	void draw() {

		//glClearColor(0.5, 0.5, 0.5, 1);
		glClearColor(1, 1, 1, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glMatrixMode(GL_MODELVIEW);
		gluPerspective(180, 1.5, -1000, 1000);
		glLoadIdentity();
		control.gx = mTrackingManager->mInputManager->mInputData.image_data.hand.center(0);
		control.gy = mTrackingManager->mInputManager->mInputData.image_data.hand.center(1);
		control.gz = mTrackingManager->mInputManager->mInputData.image_data.hand.center(2);

		//���ֵ�Ǹ���palm_Center���õģ���Ϊ���ʹ��palm_Center�Ļ����������÷ǳ�����
		if (mTrackingManager->mRuntimeType == REALTIME)
		{
			control.gx = 90;
			control.gy = -15;
			control.gz = 450;
		}

		double r =250;
		double x = r*cos(control.roty)*sin(control.rotx);
		double y = r*sin(control.roty);
		double z = r*cos(control.roty)*cos(control.rotx);
		//cout<< x <<" "<< y <<" " << z<<endl;
		gluLookAt(x + control.gx, y + control.gy, z + control.gz, control.gx, control.gy, control.gz, 0.0, 1.0, 0.0);//��������ʼ�ǿ���-z�ģ�֮��ĽǶ�����global�����ϵ��ӵģ�����Ҫ��

		draw_HandModel();
		draw_Interacted_Object();
		//draw_HandPointCloud();
		//draw_HandPointCloudNormal();
		//draw_ObjectCloud();
		//draw_ObjectCloudNormal();
		draw_Coordinate();

		glFlush();
		glutSwapBuffers();
	}

	void idle() {
		if (!pause)
		{
			mTrackingManager->Tracking(track);
			mTrackingManager->ShowRenderAddColor();
		}
		glutPostRedisplay();
	}


	//GL��ʼ������
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
		// ��ʼ��GLUT
		glutInit(&argc, argv);

		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
		glutInitWindowSize(800, 600);
		glutInitWindowPosition(100, 100);
		glutCreateWindow("SHOW RESULT");
		InitializeGlutCallbacks();
		initScene(800, 600);
	}

	void start() {
		// ֪ͨ��ʼGLUT���ڲ�ѭ��
		glutMainLoop();
	}
}