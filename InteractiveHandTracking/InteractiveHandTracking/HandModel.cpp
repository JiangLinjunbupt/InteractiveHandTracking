#include"HandModel.h"

HandModel::HandModel(Camera* _camera) :camera(_camera)
{
	this->LoadModel();
	NumOfCollision = 0;

	//设置每根手指对应的顶点
	{
		Finger_Tip[3] = 320;
		Finger_Tip[6] = 443;
		Finger_Tip[9] = 693;
		Finger_Tip[12] = 555;
		Finger_Tip[15] = 745;
	}

	//设置关节点之间的关系
	{
		//设置父节点关系
		this->Parent[0] = -1;
		for (int i = 1; i < this->Kintree_table.cols(); ++i) this->Parent[i] = this->Kintree_table(0, i);
		//设置子节点关系，为了后续计算局部坐标系
		{
			this->Child[0] = 4;

			this->Child[1] = 2;
			this->Child[2] = 3;
			this->Child[3] = -1;

			this->Child[4] = 5;
			this->Child[5] = 6;
			this->Child[6] = -1;

			this->Child[7] = 8;
			this->Child[8] = 9;
			this->Child[9] = -1;

			this->Child[10] = 11;
			this->Child[11] = 12;
			this->Child[12] = -1;

			this->Child[13] = 14;
			this->Child[14] = 15;
			this->Child[15] = -1;
		}

		//设置运动学链相关的节点关系，并且顺序按照：手腕->该节点
		this->joint_relation.resize(this->Joints_num);
		for (int i = 0; i < this->Joints_num; ++i)
		{
			std::vector<int> tmp;
			tmp.push_back(i);

			int parent = this->Parent[i];
			while (parent != -1)
			{
				tmp.push_back(parent);
				parent = this->Parent[parent];
			}
			std::sort(tmp.begin(), tmp.end());
			this->joint_relation[i] = tmp;
		}
		//for (int i = 0; i < this->Joints_num; ++i)
		//{
		//	for (auto j : this->joint_relation[i]) std::cout << j << "  ";
		//       std::cout << std::endl;
		//}
	}

	this->Local_Coordinate_Init();

	//控制手模位置、手型、状态的参数，初始都设置为0
	this->Shape_params = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
	this->Pose_params = Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS);

	//初始化顶点矩阵
	this->V_shaped = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->V_posed = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->V_Final = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	this->V_Normal_Final = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	//初始化面法向量矩阵
	this->F_normal = Eigen::MatrixXf::Zero(this->Face_num, 3);
	//初始化关节点相关矩阵
	this->J_shaped = Eigen::MatrixXf::Zero(this->Joints_num, 3);
	this->J_Final = Eigen::MatrixXf::Zero(this->Joints_num, 3);

	//初始化jacobain计算相关的
	//pose相关的
	{
		std::vector<Eigen::Matrix4f> each_joint_pose_jacobain(NUM_HAND_WRIST_PARAMS + NUM_HAND_FINGER_PARAMS, Eigen::Matrix4f::Zero());
		this->pose_jacobain_matrix.resize(this->Joints_num);
		for (int i = 0; i < this->Joints_num; ++i) this->pose_jacobain_matrix[i] = each_joint_pose_jacobain;
	}
	//shape相关的
	{
		std::vector<Eigen::Matrix4f> each_joint_shape_jacobain(NUM_HAND_SHAPE_PARAMS, Eigen::Matrix4f::Zero());
		this->shape_jacobain_matrix.resize(this->Joints_num);
		for (int i = 0; i < this->Joints_num; ++i) this->shape_jacobain_matrix[i] = each_joint_shape_jacobain;
	}
	//Trans相关的
	{
		this->Trans_child_to_parent.resize(this->Joints_num);
		this->Trans_child_to_parent_Jacob.resize(this->Joints_num);
		this->Trans_world_to_local.resize(this->Joints_num);
		this->Trans_world_to_local_Jacob.resize(this->Joints_num);

		this->result_shape_jacob.resize(this->Joints_num);

		std::vector<Eigen::Matrix4f> each_joint_Trans_jacobain(NUM_HAND_SHAPE_PARAMS, Eigen::Matrix4f::Zero());
		for (int i = 0; i < this->Joints_num; ++i)
		{
			this->Trans_child_to_parent[i] = Eigen::Matrix4f::Zero();
			this->Trans_child_to_parent_Jacob[i] = each_joint_Trans_jacobain;

			this->Trans_world_to_local[i] = Eigen::Matrix4f::Zero();
			this->Trans_world_to_local_Jacob[i] = each_joint_Trans_jacobain;

			this->result_shape_jacob[i] = each_joint_Trans_jacobain;
		}
	}

	//初始化中间变量
	this->result.resize(this->Joints_num);
	this->result2.resize(this->Joints_num);
	this->T.resize(this->Vertex_num);
	//对手模进行初始化变换

	Eigen::VectorXf pose_params_tmp = Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS);
	Eigen::VectorXf shape_params_tmp = Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS);
	pose_params_tmp.tail(NUM_HAND_FINGER_PARAMS) = this->Hands_mean;
	this->set_Shape_Params(Eigen::VectorXf::Zero(NUM_HAND_SHAPE_PARAMS));
	this->set_Pose_Params(Eigen::VectorXf::Zero(NUM_HAND_POSE_PARAMS));
	this->UpdataModel();

	std::cout << "Model Init Successed\n";
}


#pragma region LoadFunctions
void HandModel::LoadModel()
{
	std::string J_filename = ".\\model\\J.txt";
	std::string J_regressor_filename = ".\\model\\J_regressor.txt";
	std::string f_filename = ".\\model\\face.txt";
	std::string kintree_table_filename = ".\\model\\kintree_table.txt";
	std::string posedirs_filename = ".\\model\\posedirs.txt";
	std::string shapedirs_filename = ".\\model\\shapedirs.txt";
	std::string v_template_filename = ".\\model\\v_template.txt";
	std::string weights_filename = ".\\model\\weights.txt";

	std::string hands_coeffs_filename = ".\\Data\\PoseParams_afterConvert.txt";
	std::string hands_components_filename = ".\\Data\\PoseParams_eigenVec.txt";
	std::string hands_mean_filename = ".\\Data\\PoseParamsMean.txt";
	std::string hand_Pose_var_filename = ".\\Data\\PoseParams_eigenVal.txt";
	std::string hand_shape_var_filename = ".\\Data\\ShapeParams_var.txt";
	std::string hand_pose_Max_Min_filename = ".\\Data\\PoseParams_maxAndmin.txt";

	std::string gloveDifference_MaxMinPCA_filename = ".\\Data\\gloveParamsDifference_maxAndmin_PCA.txt";
	std::string gloveDifference_VarPCA_filename = ".\\Data\\gloveParamsDifference_Var_PCA.txt";

	this->Load_J(J_filename.c_str());
	this->Load_J_regressor(J_regressor_filename.c_str());
	this->Load_F(f_filename.c_str());
	this->Load_Kintree_table(kintree_table_filename.c_str());
	this->Load_Posedirs(posedirs_filename.c_str());
	this->Load_Shapedirs(shapedirs_filename.c_str());
	this->Load_V_template(v_template_filename.c_str());
	this->Load_Weights(weights_filename.c_str());

	this->Load_Hands_coeffs(hands_coeffs_filename.c_str());
	this->Load_Hands_components(hands_components_filename.c_str());
	this->Load_Hands_mean(hands_mean_filename.c_str());
	this->Load_Hand_Pose_var(hand_Pose_var_filename.c_str());
	this->Load_Hand_Shape_var(hand_shape_var_filename.c_str());
	this->Load_Hand_Pose_Max_Min(hand_pose_Max_Min_filename.c_str());

	this->Load_GloveDifference_Max_MinPCA(gloveDifference_MaxMinPCA_filename.c_str());
	this->Load_GloveDifference_VarPCA(gloveDifference_VarPCA_filename.c_str());

	std::cout << "Load Model success " << std::endl;
}

void HandModel::Load_J(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  J error,  can not open this file !!! \n";

	f >> this->Joints_num;
	this->J = Eigen::MatrixXf::Zero(this->Joints_num, 3);
	for (int i = 0; i < this->Joints_num; ++i)
	{
		f >> this->J(i, 0) >> this->J(i, 1) >> this->J(i, 2);
	}
	f.close();

	this->J = this->J * 1000;
	std::cout << "Load J success\n";
}
void HandModel::Load_J_regressor(const char* filename)
{
	//这里注意加载的是稀疏矩阵，参考：
	//https://my.oschina.net/cvnote/blog/166980   或者  https://blog.csdn.net/xuezhisdc/article/details/54631490
	//Sparse Matrix转Dense Matrix ： MatrixXd dMat; dMat = MatrixXd(spMat);  可以方便观察
	//Dense Matrix转Sparse Matrix : SparseMatrix<double> spMat; spMat = dMat.sparseView(); 

	//这里使用的是coo结构的稀疏矩阵，稀疏矩阵的类型参考：https://www.cnblogs.com/xbinworld/p/4273506.html
	Joint_related_Vertex.clear();
	for (int i = 0; i < this->Joints_num; ++i)
		Joint_related_Vertex.push_back(std::map<float, int>());


	std::vector < Eigen::Triplet <float> > triplets;

	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  J_regressor  error,  can not open this file !!! \n";

	int rows, cols, NNZ_num;
	f >> rows >> cols >> NNZ_num;
	this->J_regressor.resize(rows, cols);

	float row, col, value;
	for (int i = 0; i < NNZ_num; ++i)
	{
		f >> row >> col >> value;
		triplets.emplace_back(row, col, value);

		Joint_related_Vertex[row].insert(std::make_pair(value, col));
	}

	this->J_regressor.setFromTriplets(triplets.begin(), triplets.end());
	f.close();

	std::cout << "Load J_regressor success\n";
}
void HandModel::Load_F(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Face  error,  can not open this file !!! \n";

	f >> this->Face_num;
	this->F = Eigen::MatrixXi::Zero(this->Face_num, 3);

	for (int i = 0; i < this->Face_num; ++i)
	{
		int index1, index2, index3;
		f >> index1 >> index2 >> index3;
		this->F(i, 0) = index1;
		this->F(i, 1) = index2;
		this->F(i, 2) = index3;
	}
	f.close();

	std::cout << "Load face success\n";
}
void HandModel::Load_Kintree_table(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Kintree_table  error,  can not open this file !!! \n";

	int rows, cols;
	f >> rows >> cols;
	this->Kintree_table = Eigen::MatrixXf::Zero(rows, cols);
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			f >> this->Kintree_table(row, col);
		}
	}
	f.close();

	std::cout << "Load Kintree_table success\n";
}
void HandModel::Load_Posedirs(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Posedirs  error,  can not open this file !!! \n";

	int dim1, dim2, dim3;
	f >> dim1 >> dim2 >> dim3;
	this->Posedirs.resize(dim3);
	for (int d3 = 0; d3 < dim3; ++d3)
	{
		Eigen::MatrixXf tem = Eigen::MatrixXf::Zero(dim1, dim2);
		for (int d1 = 0; d1 < dim1; ++d1)
		{
			for (int d2 = 0; d2 < dim2; ++d2)
			{
				f >> tem(d1, d2);
			}
		}
		this->Posedirs[d3] = tem * 1000;
	}

	f.close();

	std::cout << "Load Posedirs success\n";
}
void HandModel::Load_Shapedirs(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Shapedirs  error,  can not open this file !!! \n";

	int dim1, dim2, dim3;
	f >> dim1 >> dim2 >> dim3;
	this->Shapedirs.resize(dim3);
	this->Joint_Shapedir.resize(dim3);

	for (int d3 = 0; d3 < dim3; ++d3)
	{
		Eigen::MatrixXf tem = Eigen::MatrixXf::Zero(dim1, dim2);
		for (int d1 = 0; d1 < dim1; ++d1)
		{
			for (int d2 = 0; d2 < dim2; ++d2)
			{
				f >> tem(d1, d2);
			}
		}
		this->Shapedirs[d3] = tem * 1000;

		//这里通过Shape_dir和J_regressor计算出Joint_shape_dir
		this->Joint_Shapedir[d3] = this->J_regressor * this->Shapedirs[d3];
	}

	f.close();

	std::cout << "Load Shapedirs success\n";
}
void HandModel::Load_V_template(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  V_template  error,  can not open this file !!! \n";

	int dim;
	f >> this->Vertex_num >> dim;
	this->V_template = Eigen::MatrixXf::Zero(this->Vertex_num, 3);
	for (int v = 0; v < this->Vertex_num; ++v)
	{
		f >> this->V_template(v, 0) >> this->V_template(v, 1) >> this->V_template(v, 2);
	}
	f.close();

	this->V_template = this->V_template * 1000;

	//这里为了保险，我重新计算一下V_template对应的Joint值；
	this->J = this->J_regressor * this->V_template;
	std::cout << "Load V_tempalte success\n";
}
void HandModel::Load_Weights(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  V_template  error,  can not open this file !!! \n";

	int rows, cols;
	f >> rows >> cols;
	this->Weights = Eigen::MatrixXf::Zero(rows, cols);
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			f >> this->Weights(row, col);
		}
	}
	f.close();

	std::cout << "Load Weights success\n";
}


void HandModel::Load_Hands_coeffs(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Hands_coeffs  error,  can not open this file !!! \n";

	int rows, cols;
	f >> rows >> cols;
	this->Hands_coeffs = Eigen::MatrixXf::Zero(rows, cols);
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			f >> this->Hands_coeffs(row, col);
		}
	}
	f.close();

	std::cout << "Load Hands_coeffs success\n";
}
void HandModel::Load_Hands_components(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Hands_components  error,  can not open this file !!! \n";

	int rows, cols;
	f >> rows >> cols;
	this->Hands_components = Eigen::MatrixXf::Zero(rows, cols);
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			f >> this->Hands_components(row, col);
		}
	}
	f.close();
	std::cout << "Load Hands_components success\n";
}
void HandModel::Load_Hands_mean(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Hands_mean  error,  can not open this file !!! \n";

	int num;
	f >> num;
	this->Hands_mean = Eigen::VectorXf::Zero(num);
	for (int i = 0; i < num; ++i)
		f >> this->Hands_mean(i);
	f.close();

	std::cout << "Load Hands_mean success\n";
}
void HandModel::Load_Hand_Pose_var(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  HandPoseVar  error,  can not open this file !!! \n";

	int num;
	f >> num;
	this->Hand_Pose_var = Eigen::VectorXf::Zero(num);
	for (int i = 0; i < num; ++i)
		f >> this->Hand_Pose_var(i);
	f.close();

	std::cout << "Load HandPoseVar success\n";
}
void HandModel::Load_Hand_Shape_var(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  HandShapeVar  error,  can not open this file !!! \n";

	int num;
	f >> num;
	this->Hand_Shape_var = Eigen::VectorXf::Zero(num);
	for (int i = 0; i < num; ++i)
		f >> this->Hand_Shape_var(i);
	f.close();

	std::cout << "Load HandShapeVar success\n";
}
void HandModel::Load_Hand_Pose_Max_Min(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  Hands_Pose_MaxMin  error,  can not open this file !!! \n";

	int num;
	f >> num;
	this->Hands_Pose_Max = Eigen::VectorXf::Zero(num);
	this->Hands_Pose_Min = Eigen::VectorXf::Zero(num);

	for (int i = 0; i < num; ++i)
		f >> this->Hands_Pose_Max(i) >> this->Hands_Pose_Min(i);
	f.close();

	std::cout << "Load Hands_Pose_MaxMin success\n";


	for (int i = 0; i < num; ++i)
	{
		std::cout << " max is :  " << this->Hands_Pose_Max(i) << "   Min  is :  " << this->Hands_Pose_Min(i) << std::endl;
	}
}

void HandModel::Load_GloveDifference_Max_MinPCA(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  GloveDifference_Max_Min_PCA  error,  can not open this file !!! \n";

	f >> this->K_PCAcomponnet;
	this->Glove_Difference_MinPCA = Eigen::VectorXf::Zero(this->K_PCAcomponnet);
	this->Glove_Difference_MaxPCA = Eigen::VectorXf::Zero(this->K_PCAcomponnet);

	for (int i = 0; i < this->K_PCAcomponnet; ++i)
		f >> this->Glove_Difference_MaxPCA(i) >> this->Glove_Difference_MinPCA(i);
	f.close();

	std::cout << "Load GloveDifference_Max_Min_PCA success\n";
}
void HandModel::Load_GloveDifference_VarPCA(const char* filename)
{
	std::ifstream f;
	f.open(filename);
	if (!f.is_open())  std::cerr << "Load  GloveDifference_var_PCA  error,  can not open this file !!! \n";

	int num;
	f >> num;
	if (num != this->K_PCAcomponnet) std::cerr << "K_PCAComponnet Error!!!\n";

	this->Glove_Difference_VARPCA = Eigen::VectorXf::Zero(num);
	for (int i = 0; i < num; ++i)
		f >> this->Glove_Difference_VARPCA(i);
	f.close();

	std::cout << "Load GloveDifference_var_PCA success\n";
}
#pragma endregion LoadFunctions


void HandModel::UpdataModel()
{
	auto m_begin = high_resolution_clock::now();
	this->Updata_V_rest();
	this->LBS_Updata();
	this->NormalUpdata();
	this->Jacob_Matrix_Updata();

	this->Judge_Collision();

	//cout << "模型变换需要的时间 ： " << duration_cast<chrono::microseconds>(high_resolution_clock::now() - m_begin).count() << " us\n";
}

void HandModel::Updata_V_rest()
{
	if (want_shapemodel && change_shape)
	{
		this->ShapeSpaceBlend();
		//更新一次形状，重新求一次初始状态下的碰撞体
		this->Compute_JointRadius();
		this->Compute_CollisionSphere();

	}
	this->PoseSpaceBlend();
}
void HandModel::ShapeSpaceBlend()
{
	this->V_shaped = this->V_template;
	this->J_shaped = this->J;

	for (int i = 0; i < NUM_HAND_SHAPE_PARAMS; ++i)
	{
		this->V_shaped += this->Shapedirs[i] * Shape_params[i];
		this->J_shaped += this->Joint_Shapedir[i] * Shape_params[i];
	}

	//为了后续的雅各比计算，这句话被我改编成了：在加载过程中先计算Joint_Shapedir，再通过Shape_params控制Joint的变化。
	//this->J_shaped = this->J_regressor*this->V_shaped;
	this->Local_Coordinate_Updata();
	this->Trans_Matrix_Updata();
	this->change_shape = false;
}

void HandModel::Local_Coordinate_Init()
{
	this->Local_Coordinate.resize(this->Joints_num);
	for (int i = 0; i < this->Joints_num; ++i) this->Local_Coordinate[i] = Eigen::Matrix4f::Zero();

	Eigen::Vector3f axis_x, axis_y, axis_z;

	for (int i = 0; i < this->Joints_num; ++i)
	{
		if (this->Child[i] != -1)
		{
			axis_x(0) = this->J(i, 0) - this->J(this->Child[i], 0);
			axis_x(1) = this->J(i, 1) - this->J(this->Child[i], 1);
			axis_x(2) = this->J(i, 2) - this->J(this->Child[i], 2);

			axis_z << 0.0f, 0.0f, 1.0f;

			//y = z*x
			axis_x.normalize();
			axis_y = axis_z.cross(axis_x);
			//z = x*y
			axis_y.normalize();
			axis_z = axis_x.cross(axis_y);
			axis_z.normalize();

			this->Local_Coordinate[i](0, 0) = axis_x(0); this->Local_Coordinate[i](0, 1) = axis_y(0); this->Local_Coordinate[i](0, 2) = axis_z(0); this->Local_Coordinate[i](0, 3) = this->J(i, 0);
			this->Local_Coordinate[i](1, 0) = axis_x(1); this->Local_Coordinate[i](1, 1) = axis_y(1); this->Local_Coordinate[i](1, 2) = axis_z(1); this->Local_Coordinate[i](1, 3) = this->J(i, 1);
			this->Local_Coordinate[i](2, 0) = axis_x(2); this->Local_Coordinate[i](2, 1) = axis_y(2); this->Local_Coordinate[i](2, 2) = axis_z(2); this->Local_Coordinate[i](2, 3) = this->J(i, 2);
			this->Local_Coordinate[i](3, 0) = 0.0f;      this->Local_Coordinate[i](3, 1) = 0.0f;      this->Local_Coordinate[i](3, 2) = 0.0f;      this->Local_Coordinate[i](3, 3) = 1.0f;
		}
		else
		{
			this->Local_Coordinate[i] = this->Local_Coordinate[this->Parent[i]];
			this->Local_Coordinate[i](0, 3) = this->J(i, 0);
			this->Local_Coordinate[i](1, 3) = this->J(i, 1);
			this->Local_Coordinate[i](2, 3) = this->J(i, 2);
		}
	}
}
void HandModel::Local_Coordinate_Updata()
{
	for (int i = 0; i < this->Joints_num; ++i)
	{
		this->Local_Coordinate[i](0, 3) = this->J_shaped(i, 0);
		this->Local_Coordinate[i](1, 3) = this->J_shaped(i, 1);
		this->Local_Coordinate[i](2, 3) = this->J_shaped(i, 2);
	}
}
void HandModel::Trans_Matrix_Updata()
{
	//因为0节点的父节点是世界坐标系，比较特殊，所以先设置0节点的值
	this->Trans_child_to_parent[0] = this->Local_Coordinate[0];

	Eigen::Matrix4f Trans_0_to_world_jacob = Eigen::Matrix4f::Zero();

	for (int i = 0; i < NUM_HAND_SHAPE_PARAMS; ++i)
	{
		Trans_0_to_world_jacob.block(0, 3, 3, 1) = (this->Joint_Shapedir[i].row(0)).transpose();

		this->Trans_child_to_parent_Jacob[0][i] = Trans_0_to_world_jacob;
	}

	for (int i = 1; i < this->Joints_num; ++i)
	{
		this->Trans_child_to_parent[i] = this->Local_Coordinate[this->Parent[i]].inverse()*this->Local_Coordinate[i];

		for (int j = 0; j < NUM_HAND_SHAPE_PARAMS; ++j)
		{
			//这里会用到 逆矩阵的求导
			Eigen::Matrix4f first_item = Eigen::Matrix4f::Zero();
			first_item.block(0, 3, 3, 1) = (this->Joint_Shapedir[j].row(this->Parent[i])).transpose();
			first_item = -1 * this->Local_Coordinate[this->Parent[i]].inverse()*first_item*this->Local_Coordinate[this->Parent[i]].inverse()*this->Local_Coordinate[i];

			Eigen::Matrix4f secont_item = Eigen::Matrix4f::Zero();
			secont_item.block(0, 3, 3, 1) = (this->Joint_Shapedir[j].row(i)).transpose();
			secont_item = this->Local_Coordinate[this->Parent[i]].inverse()*secont_item;

			this->Trans_child_to_parent_Jacob[i][j] = first_item + secont_item;
		}
	}

	for (int i = 0; i < this->Joints_num; ++i)
	{
		this->Trans_world_to_local[i] = this->Local_Coordinate[i].inverse();

		for (int j = 0; j < NUM_HAND_SHAPE_PARAMS; ++j)
		{
			Eigen::Matrix4f tmp = Eigen::Matrix4f::Zero();
			tmp.block(0, 3, 3, 1) = (this->Joint_Shapedir[j].row(i)).transpose();
			this->Trans_world_to_local_Jacob[i][j] = -1 * this->Local_Coordinate[i].inverse()*tmp*this->Local_Coordinate[i].inverse();
		}
	}

}
void HandModel::PoseSpaceBlend()
{
	this->V_posed = this->V_shaped;

	std::vector<float> pose_vec = lortmin(this->Pose_params.tail(NUM_HAND_FINGER_PARAMS));

	for (int i = 0; i < pose_vec.size(); ++i)
	{
		this->V_posed += this->Posedirs[i] * pose_vec[i];
	}
}

void HandModel::LBS_Updata()
{
	//这一步的意思是，我先旋转，在将旋转后的结果转换到世界坐标系下，因此是：Trans*R
	Eigen::Matrix4f Rotate_0 = Eigen::Matrix4f::Identity();
	Rotate_0.block(0, 0, 3, 3) = EularToRotateMatrix(this->Pose_params[3], this->Pose_params[4], this->Pose_params[5]);
	result[0] = this->Trans_child_to_parent[0] * Rotate_0;

	//求result[0]的关于形状参数的雅各比
	Eigen::Matrix4f tmp = Eigen::Matrix4f::Zero();
	for (int i = 0; i < NUM_HAND_SHAPE_PARAMS; ++i)
	{
		tmp = this->Trans_child_to_parent_Jacob[0][i] * Rotate_0;
		this->result_shape_jacob[0][i] = tmp;
	}

	//求剩下的result
	for (int i = 1; i < this->Joints_num; ++i)
	{
		Eigen::Matrix4f Rotate = Eigen::Matrix4f::Identity();
		Rotate.block(0, 0, 3, 3) = EularToRotateMatrix(this->Pose_params[(i + 1) * 3 + 0], this->Pose_params[(i + 1) * 3 + 1], this->Pose_params[(i + 1) * 3 + 2]);

		result[i] = result[this->Parent[i]] * this->Trans_child_to_parent[i] * Rotate;

		//计算result的jacob
		for (int j = 0; j < NUM_HAND_SHAPE_PARAMS; ++j)
		{
			tmp = (result_shape_jacob[this->Parent[i]][j] * this->Trans_child_to_parent[i] + result[this->Parent[i]] * this->Trans_child_to_parent_Jacob[i][j])*Rotate;
			this->result_shape_jacob[i][j] = tmp;
		}
	}

	//这一步的意思是，由于给出的顶点坐标和关节点坐标都是世界坐标系下的，因此我先要从世界坐标系转换到每个关节点下的局部坐标系，再做旋转变换，因此： R*Trans 
	for (int i = 0; i < result.size(); ++i)
	{
		result2[i] = result[i] * this->Trans_world_to_local[i];
	}

	//关节点变换，关节点没有权重分布，因此可以直接变换
	for (int i = 0; i < this->Joints_num; ++i)
	{
		Eigen::Vector4f temp(this->J_shaped.row(i)(0), this->J_shaped.row(i)(1), this->J_shaped.row(i)(2), 1);
		this->J_Final.row(i) = ((result2[i] * temp).head(3)).transpose() + (this->Pose_params.head(NUM_HAND_POSITION_PARAMS)).transpose();
	}

	//碰撞体变换
	{
		for (int i = 0; i < Collision_sphere.size(); ++i)
		{
			int joint_idx = this->Collision_sphere[i].joint_belong;
			Eigen::Vector4f temp(this->Collision_sphere[i].Init_Center(0), this->Collision_sphere[i].Init_Center(1), this->Collision_sphere[i].Init_Center(2), 1);

			this->Collision_sphere[i].Update_Center = (result2[joint_idx] * temp).head(3) + this->Pose_params.head(NUM_HAND_POSITION_PARAMS);
		}
	}

	//这个是考虑顶点权重累计之后的变换
	for (int i = 0; i < T.size(); ++i)
	{
		T[i].setZero();
		for (int j = 0; j < this->Joints_num; ++j)
		{
			T[i] += result2[j] * this->Weights(i, j);
		}
	}

	this->V_Final.setZero();

	for (int i = 0; i < this->Vertex_num; ++i)
	{
		Eigen::Vector4f temp(this->V_posed.row(i)(0), this->V_posed.row(i)(1), this->V_posed.row(i)(2), 1);
		this->V_Final.row(i) = ((T[i] * temp).head(3)).transpose() + (this->Pose_params.head(NUM_HAND_POSITION_PARAMS)).transpose();
	}

}
void HandModel::NormalUpdata()
{
	this->V_Normal_Final.setZero();
	this->F_normal.setZero();

	//float max_triangle_area = 0.0f;
	//float average_triangle_area = 0.0f;

	for (int i = 0; i < this->Face_num; ++i)
	{
		Eigen::Vector3f A, B, C, BA, BC;
		//这里我假设，如果假设错了，那么叉乘时候，就BC*BA变成BA*BC
		//            A
		//          /  \
												       //         B — C
		A << this->V_Final(this->F(i, 0), 0), this->V_Final(this->F(i, 0), 1), this->V_Final(this->F(i, 0), 2);
		B << this->V_Final(this->F(i, 1), 0), this->V_Final(this->F(i, 1), 1), this->V_Final(this->F(i, 1), 2);
		C << this->V_Final(this->F(i, 2), 0), this->V_Final(this->F(i, 2), 1), this->V_Final(this->F(i, 2), 2);


		BC << C - B;
		BA << A - B;

		//float area = (BC.cross(BA)).norm();
		//average_triangle_area += area;
		//if (area > max_triangle_area)
		//	max_triangle_area = area;

		Eigen::Vector3f nom(BC.cross(BA));

		nom.normalize();

		this->V_Normal_Final(this->F(i, 0), 0) += nom(0);
		this->V_Normal_Final(this->F(i, 0), 1) += nom(1);
		this->V_Normal_Final(this->F(i, 0), 2) += nom(2);

		this->V_Normal_Final(this->F(i, 1), 0) += nom(0);
		this->V_Normal_Final(this->F(i, 1), 1) += nom(1);
		this->V_Normal_Final(this->F(i, 1), 2) += nom(2);

		this->V_Normal_Final(this->F(i, 2), 0) += nom(0);
		this->V_Normal_Final(this->F(i, 2), 1) += nom(1);
		this->V_Normal_Final(this->F(i, 2), 2) += nom(2);


		this->F_normal(i, 0) = nom(0);
		this->F_normal(i, 1) = nom(1);
		this->F_normal(i, 2) = nom(2);
	}

	V_Visible_2D.clear();
	int width = camera->width();
	int height = camera->height();

	for (int i = 0; i < this->Vertex_num; ++i)
	{
		this->V_Normal_Final.row(i).normalize();

		if (this->V_Normal_Final(i, 2) < 0)
		{
			Eigen::Vector2f idx_2D = camera->world_to_image(Eigen::Vector3f(this->V_Final(i, 0), this->V_Final(i, 1), this->V_Final(i, 2)));

			int row_ = (int)idx_2D(1);
			int col_ = (int)idx_2D(0);

			if (row_ >= 0 && row_ <height && col_ >= 0 && col_ < width)
			{
				V_Visible_2D.push_back(make_pair(Eigen::Vector2i(col_, row_), i));
			}
		}
	}

	//cout << "面积最大的三角形的面积为： " << max_triangle_area << "   平均面积为： " << average_triangle_area / this->Face_num << endl;
}

void HandModel::Save_as_obj()
{
	std::ofstream f;
	f.open("MANO_HandModel.obj");
	if (!f.is_open())  std::cout << "Can not Save to .obj file, The file can not open ！！！\n";

	for (int i = 0; i < this->Vertex_num; ++i)
	{
		f << "v " << this->V_Final(i, 0) << " " << this->V_Final(i, 1) << " " << this->V_Final(i, 2) << std::endl;
	}

	for (int i = 0; i < this->Face_num; ++i)
	{
		f << "f " << (this->F(i, 0) + 1) << " " << (this->F(i, 1) + 1) << " " << (this->F(i, 2) + 1) << std::endl;
	}
	f.close();
	std::cout << "Save to MANO_HandModel.obj success\n";
}


void HandModel::Jacob_Matrix_Updata()
{
	this->Pose_Jacobain_matrix_Updata();
	this->Shape_Jacobain_matrix_Updata();
}
void HandModel::Pose_Jacobain_matrix_Updata()
{
	Eigen::Matrix4f x_jacob, y_jacob, z_jacob;

	for (int i = 0; i < this->Joints_num; ++i)
	{
		//先清零
		for (int j = 0; j < this->pose_jacobain_matrix[i].size(); ++j) this->pose_jacobain_matrix[i][j].setZero();

		//对于树状结构的那部分，再进行计算
		for (int j = 0; j < this->joint_relation[i].size(); ++j)
		{
			int index = this->joint_relation[i][j];

			this->RotateMatrix_jacobain(this->Pose_params[(index + 1) * 3 + 0], this->Pose_params[(index + 1) * 3 + 1], this->Pose_params[(index + 1) * 3 + 2],
				x_jacob, y_jacob, z_jacob);


			if (index == 0)
			{
				x_jacob = this->Trans_child_to_parent[0] * x_jacob;
				y_jacob = this->Trans_child_to_parent[0] * y_jacob;
				z_jacob = this->Trans_child_to_parent[0] * z_jacob;

			}
			else
			{
				x_jacob = this->result[this->Parent[index]] * this->Trans_child_to_parent[index] * x_jacob;
				y_jacob = this->result[this->Parent[index]] * this->Trans_child_to_parent[index] * y_jacob;
				z_jacob = this->result[this->Parent[index]] * this->Trans_child_to_parent[index] * z_jacob;
			}


			if (j < this->joint_relation[i].size() - 1)
			{
				int z = this->joint_relation[i][j + 1];
				for (; z <= this->joint_relation[i].back(); ++z)
				{

					Eigen::Matrix4f Rotate = Eigen::Matrix4f::Identity();
					Rotate.block(0, 0, 3, 3) = EularToRotateMatrix(this->Pose_params[(z + 1) * 3 + 0], this->Pose_params[(z + 1) * 3 + 1], this->Pose_params[(z + 1) * 3 + 2]);

					x_jacob = x_jacob*this->Trans_child_to_parent[z] * Rotate;
					y_jacob = y_jacob*this->Trans_child_to_parent[z] * Rotate;
					z_jacob = z_jacob*this->Trans_child_to_parent[z] * Rotate;
				}
			}

			this->pose_jacobain_matrix[i][index * 3 + 0] = x_jacob*this->Trans_world_to_local[i];
			this->pose_jacobain_matrix[i][index * 3 + 1] = y_jacob*this->Trans_world_to_local[i];
			this->pose_jacobain_matrix[i][index * 3 + 2] = z_jacob*this->Trans_world_to_local[i];

		}
	}
}
void HandModel::Shape_Jacobain_matrix_Updata()
{
	for (int i = 0; i < this->Joints_num; ++i)
	{
		for (int j = 0; j < NUM_HAND_SHAPE_PARAMS; ++j)
		{
			this->shape_jacobain_matrix[i][j] = this->result_shape_jacob[i][j] * this->Trans_world_to_local[i] + this->result[i] * this->Trans_world_to_local_Jacob[i][j];
		}
	}
}
//两个雅各比函数
void HandModel::Shape_jacobain(Eigen::MatrixXf& jacobain, int vertex_id, const Eigen::Vector3f& vertex_pos)
{
	jacobain = Eigen::MatrixXf::Zero(3, NUM_HAND_SHAPE_PARAMS);

	//for (int i = 0; i < this->Shape_params_num; ++i)
	//{
	//	//这里的齐次坐标表示为（Δx,Δy,Δz,0)的原因是, v_shaped = (x + Δx[i], y + Δy[i], z + Δz[i], 1 ) = (x, y , z , 1 ) + （Δx,Δy,Δz,0);
	//	Eigen::Vector4f temp(this->Shapedirs[i](vertex_id, 0), this->Shapedirs[i](vertex_id, 1), Shapedirs[i](vertex_id, 2), 0);
	//	jacobain.col(i) = (T[vertex_id] * temp).head(3);
	//	//jacobain(0, i) = this->Shapedirs[i](vertex_id, 0);
	//	//jacobain(1, i) = this->Shapedirs[i](vertex_id, 1);
	//	//jacobain(2, i) = this->Shapedirs[i](vertex_id, 2);
	//}

	Eigen::Vector4f temp_2;

	if (vertex_pos == Eigen::Vector3f::Zero())
		temp_2 << this->V_posed.row(vertex_id)(0), this->V_posed.row(vertex_id)(1), this->V_posed.row(vertex_id)(2), 1;
	else
		temp_2 << vertex_pos(0), vertex_pos(1), vertex_pos(2), 1;

	//Eigen::Vector4f temp_2(this->V_posed.row(vertex_id)(0), this->V_posed.row(vertex_id)(1), this->V_posed.row(vertex_id)(2), 1);

	for (int i = 0; i < NUM_HAND_SHAPE_PARAMS; ++i)
	{
		Eigen::Vector4f temp(this->Shapedirs[i](vertex_id, 0), this->Shapedirs[i](vertex_id, 1), Shapedirs[i](vertex_id, 2), 0);
		Eigen::Matrix4f T_jacob = Eigen::Matrix4f::Zero();
		for (int j = 0; j < this->Joints_num; ++j)
		{
			T_jacob += this->Weights(vertex_id, j) * this->shape_jacobain_matrix[j][i];
		}
		jacobain.col(i) = (T_jacob*temp_2 + T[vertex_id] * temp).head(3);
		//jacobain.col(i) = (T[vertex_id] * temp).head(3);
	}
}

void HandModel::Pose_jacobain(Eigen::MatrixXf& jacobain, int vertex_id, const Eigen::Vector3f& vertex_pos)
{
	jacobain = Eigen::MatrixXf::Zero(3, NUM_HAND_POSE_PARAMS);

	int v_idx = vertex_id;

	//Eigen::Vector4f temp(this->V_posed.row(v_idx)(0), this->V_posed.row(v_idx)(1), this->V_posed.row(v_idx)(2), 1);
	Eigen::Vector4f temp;
	if (vertex_pos == Eigen::Vector3f::Zero())
		temp << this->V_posed.row(v_idx)(0), this->V_posed.row(v_idx)(1), this->V_posed.row(v_idx)(2), 1;
	else
		temp << vertex_pos(0), vertex_pos(1), vertex_pos(2), 1;

	//因为头三个参数是全局位移
	jacobain.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();

	Eigen::Matrix4f T_x = Eigen::Matrix4f::Zero();
	Eigen::Matrix4f T_y = Eigen::Matrix4f::Zero();
	Eigen::Matrix4f T_z = Eigen::Matrix4f::Zero();

	for (int i = 0; i < this->Joints_num; ++i)
	{
		T_x.setZero();
		T_y.setZero();
		T_z.setZero();

		for (int j = 0; j < this->Joints_num; ++j)
		{
			T_x += this->pose_jacobain_matrix[j][i * 3 + 0] * this->Weights(v_idx, j);
			T_y += this->pose_jacobain_matrix[j][i * 3 + 1] * this->Weights(v_idx, j);
			T_z += this->pose_jacobain_matrix[j][i * 3 + 2] * this->Weights(v_idx, j);
		}

		jacobain.col((i + 1) * 3 + 0) = (T_x*temp).head(3);
		jacobain.col((i + 1) * 3 + 1) = (T_y*temp).head(3);
		jacobain.col((i + 1) * 3 + 2) = (T_z*temp).head(3);

	}

}

void HandModel::Joint_Pose_jacobain(Eigen::MatrixXf& jacobain, int joint_id)
{
	jacobain = Eigen::MatrixXf::Zero(3, NUM_HAND_POSE_PARAMS);

	int v_idx = joint_id;
	Eigen::Vector4f temp(this->J_shaped(v_idx, 0), this->J_shaped(v_idx, 1), this->J_shaped(v_idx, 2), 1.0f);

	//因为头三个参数是全局位移
	jacobain.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();

	Eigen::Matrix4f T_x = Eigen::Matrix4f::Zero();
	Eigen::Matrix4f T_y = Eigen::Matrix4f::Zero();
	Eigen::Matrix4f T_z = Eigen::Matrix4f::Zero();

	for (int i = 0; i < this->Joints_num; ++i)
	{
		jacobain.col((i + 1) * 3 + 0) = (this->pose_jacobain_matrix[v_idx][i * 3 + 0] * temp).head(3);
		jacobain.col((i + 1) * 3 + 1) = (this->pose_jacobain_matrix[v_idx][i * 3 + 1] * temp).head(3);
		jacobain.col((i + 1) * 3 + 2) = (this->pose_jacobain_matrix[v_idx][i * 3 + 2] * temp).head(3);

	}

}



//**************************碰撞相关*******************************************
void HandModel::Compute_JointRadius()
{
	Joint_Radius.clear();

	for (int i = 1; i < 16; ++i)
	{
		std::set<float> tmp;  //用于自动排序

		int j = 0;
		for (std::map<float, int>::reverse_iterator itr = this->Joint_related_Vertex[i].rbegin(); j < 20; ++j, ++itr)
		{
			int idx = itr->second;

			float distance = (this->J_shaped.row(i) - this->V_shaped.row(idx)).norm();

			tmp.insert(distance);
		}

		Joint_Radius[i] = *(tmp.begin());
	}
}
void HandModel::Compute_CollisionSphere()
{
	int id_count = 0;

	Collision_sphere.clear();

	for (int i = 1; i < 16; ++i)
	{
		FingerType fingertype;

		if (i == 1 || i == 2 || i == 3) fingertype = Index;
		if (i == 4 || i == 5 || i == 6) fingertype = Middle;
		if (i == 7 || i == 8 || i == 9) fingertype = Pinky;
		if (i == 10 || i == 11 || i == 12) fingertype = Ring;
		if (i == 13 || i == 14 || i == 15) fingertype = Thumb;

		{
			//然后保存
			Collision coll;
			coll.id = id_count;
			coll.Init_Center = this->J_shaped.row(i).transpose();
			coll.Radius = this->Joint_Radius[i];
			coll.root = true;
			coll.joint_belong = i;
			coll.fingerType = fingertype;
			Collision_sphere.push_back(coll);

			++id_count;
		}
		//画从该节点到该节点的子节点的碰撞球，利用相似原理
		int ShpereNum = 3;
		{
			int Child_id = this->Child[i];

			if (Child_id > 0)
			{
				//计算长度和方向
				//首先是方向
				Eigen::VectorXf dir = ((this->J_shaped.row(Child_id) - this->J_shaped.row(i)).transpose()).normalized();
				float len = (this->J_shaped.row(Child_id) - this->J_shaped.row(i)).norm();

				//步长
				float step = len / (ShpereNum + 1.0f);

				for (int num = 1; num <= ShpereNum; ++num)
				{
					//球心
					Eigen::VectorXf Shpere_center = this->J_shaped.row(i).transpose() + num*step * dir;

					//半径，计算提示:梯形的相似比
					/*要求EF长度，
					画一条线，将梯形分为矩形和三角形，然后，用三角形的相似比算
					A ___________ B
					**|          \
					E |___________\  F
					**|            \
					C |_____________\ D

					A ___________ B
					**|          |\
					E |_________I|_\  F
					**|          |  \
					C |__________|___\ D
					*************G
					此时，IF = EF - AB ; GD = CD - AB; 他们可以用三角形相似计算

					*/
					float R = (len - num*step) / len * (Joint_Radius[i] - Joint_Radius[Child_id]) + Joint_Radius[Child_id];

					//当然你也可以为了简单直接使用子节点的半径
					//float R = Joint_Radius[Child_id];

					//然后保存
					Collision coll;
					coll.id = id_count;
					coll.Init_Center = Shpere_center;
					coll.Radius = R;
					coll.root = false;
					coll.joint_belong = i;
					coll.fingerType = fingertype;
					Collision_sphere.push_back(coll);
					++id_count;

				}
			}

			//这就属于指尖的部分了，对应指尖的部分的考虑是，先找到指尖的点（5个），然后同样方向，距离，半径随着距离略微降低
			if (Child_id < 0)
			{
				int vertex_id = Finger_Tip[i];

				//计算长度和方向
				//首先是方向
				Eigen::VectorXf dir = ((this->V_shaped.row(vertex_id) - this->J_shaped.row(i)).transpose()).normalized();
				float len = (this->V_shaped.row(vertex_id) - this->J_shaped.row(i)).norm();

				//步长
				float step = len / (ShpereNum + 1.0f);
				float radius_attuation = 0.9f;

				for (int num = 1; num <= ShpereNum; ++num)
				{
					//球心
					Eigen::VectorXf Shpere_center = this->J_shaped.row(i).transpose() + num*step * dir;
					float R = Joint_Radius[i] * pow(radius_attuation, num);

					//然后保存
					Collision coll;
					coll.id = id_count;
					coll.Init_Center = Shpere_center;
					coll.Radius = R;
					coll.root = false;
					coll.joint_belong = i;
					coll.fingerType = fingertype;
					Collision_sphere.push_back(coll);
					++id_count;
				}

			}
		}
	}

	//初始化
	int NumOfCollision = Collision_sphere.size();
	this->Collision_Judge_Matrix = Eigen::MatrixXi::Zero(NumOfCollision, NumOfCollision);
	//for (int i = 0; i < Collision_sphere.size(); ++i)
	//{
	//	cout << "id : " << Collision_sphere[i].id << '\t' << "joint_belong : " << Collision_sphere[i].joint_belong << '\t' << "radius : " << Collision_sphere[i].Radius << endl;
	//}
}
void HandModel::Judge_Collision()
{
	int NumOfCollisionSphere = Collision_sphere.size();
	this->Collision_Judge_Matrix = Eigen::MatrixXi::Zero(NumOfCollisionSphere, NumOfCollisionSphere);

	NumOfCollision = 0;
	for (int i = 0; i < NumOfCollisionSphere; ++i) {
		for (int j = 0; j < NumOfCollisionSphere; j++) {
			if (Collision_sphere[i].fingerType != Collision_sphere[j].fingerType)
			{
				float distance = (Collision_sphere[j].Update_Center - Collision_sphere[i].Update_Center).norm();

				if (distance < (Collision_sphere[j].Radius + Collision_sphere[i].Radius))
				{
					this->Collision_Judge_Matrix(i, j) = 1;
					++NumOfCollision;
				}
			}
		}
	}
}

void HandModel::CollisionPoint_Jacobian(Eigen::MatrixXf& jacobain, int jointBelong, Eigen::Vector3f& point)
{
	jacobain = Eigen::MatrixXf::Zero(3, NUM_HAND_POSE_PARAMS);

	int joint_id = jointBelong;
	Eigen::Vector4f tmp(point(0), point(1), point(2), 1.0f);

	jacobain.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();

	for (int i = 0; i < this->Joints_num; ++i)
	{
		jacobain.col((i + 1) * 3 + 0) = (this->pose_jacobain_matrix[joint_id][i * 3 + 0] * tmp).head(3);
		jacobain.col((i + 1) * 3 + 1) = (this->pose_jacobain_matrix[joint_id][i * 3 + 1] * tmp).head(3);
		jacobain.col((i + 1) * 3 + 2) = (this->pose_jacobain_matrix[joint_id][i * 3 + 2] * tmp).head(3);
	}
}


void HandModel::GenerateDepthMap()
{
	int width = camera->width();
	int heigth = camera->height();
	HandModel_depthMap = cv::Mat(cv::Size(width, heigth), CV_16UC1, cv::Scalar(0));
	HandModel_binaryMap = cv::Mat(cv::Size(width, heigth), CV_8UC1, cv::Scalar(0));

	if (J_Final(0, 2) < 200)
		return;
	for (int faceidx = 0; faceidx < this->Face_num; ++faceidx)
	{
		int a_idx = this->F(faceidx, 0);
		int b_idx = this->F(faceidx, 1);
		int c_idx = this->F(faceidx, 2);

		Eigen::Vector2f a_2D = camera->world_to_image(Eigen::Vector3f(this->V_Final(a_idx, 0), this->V_Final(a_idx, 1), this->V_Final(a_idx, 2)));
		Eigen::Vector2f b_2D = camera->world_to_image(Eigen::Vector3f(this->V_Final(b_idx, 0), this->V_Final(b_idx, 1), this->V_Final(b_idx, 2)));
		Eigen::Vector2f c_2D = camera->world_to_image(Eigen::Vector3f(this->V_Final(c_idx, 0), this->V_Final(c_idx, 1), this->V_Final(c_idx, 2)));

		int x_min = min(min(a_2D.x(), b_2D.x()), c_2D.x());
		int y_min = min(min(a_2D.y(), b_2D.y()), c_2D.y());
		int x_max = max(max(a_2D.x(), b_2D.x()), c_2D.x());
		int y_max = max(max(a_2D.y(), b_2D.y()), c_2D.y());

		//插值深度，并且进行判断
		float alpha0 = 0, alpha1 = 0;
		float depthA = 0, depthB = 0, depthC = 0;

		float d0 = 0, d1 = 0, d2 = 0;

		Eigen::Vector2f vector_aTob = b_2D - a_2D;
		Eigen::Vector2f vector_aToc = c_2D - a_2D;

		depthA = this->V_Final(a_idx, 2);
		depthB = this->V_Final(b_idx, 2);
		depthC = this->V_Final(c_idx, 2);

		for (int y = y_min; y <= y_max; ++y)
		{
			for (int x = x_min; x <= x_max; ++x)
			{
				if (x >= 0 && x < width && y >= 0 && y < heigth)
				{
					int a = (b_2D.x() - a_2D.x()) * (y - a_2D.y()) - (b_2D.y() - a_2D.y()) * (x - a_2D.x());
					int b = (c_2D.x() - b_2D.x()) * (y - b_2D.y()) - (c_2D.y() - b_2D.y()) * (x - b_2D.x());
					int c = (a_2D.x() - c_2D.x()) * (y - c_2D.y()) - (a_2D.y() - c_2D.y()) * (x - c_2D.x());

					if ((a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0))
					{
						HandModel_binaryMap.at<uchar>(y, x) = 255;

						Eigen::Vector2f p(x, y);
						Eigen::Vector2f vector_aTop = p - a_2D;
						Eigen::Vector2f vector_bTop = p - b_2D;

						//叉乘计算面积
						float S_abc = vector_aTob.x() * vector_aToc.y() - vector_aToc.x() * vector_aTob.y();
						float S_abp = vector_aTob.x() * vector_aTop.y() - vector_aTop.x() * vector_aTob.y();
						float S_apc = vector_aTop.x() * vector_aToc.y() - vector_aToc.x() * vector_aTop.y();

						if (S_abc != 0) {
							alpha1 = S_abp / S_abc;
							alpha0 = S_apc / S_abc;
						}
						else {
							//说明这些点共线

							//如果 vector_aTob 和vector_aTop方向相同
							if (vector_aTob.dot(vector_aTop) >= 0)
							{
								alpha1 = 0;
								if (vector_aTob.y() != 0) {
									alpha0 = (vector_aTop.y()) / (vector_aTob.y());
								}
								else { alpha0 = (vector_aTop.x()) / (vector_aTob.x()); }
							}
							else
							{
								alpha0 = 0;
								if (vector_aToc.y() != 0) {
									alpha1 = (vector_aTop.y()) / (vector_aToc.y());
								}
								else { alpha1 = (vector_aTop.x()) / (vector_aToc.x()); }
							}
						}

						//重心坐标系相关知识
						float depth = depthA + alpha0*(depthB - depthA) + alpha1*(depthC - depthA);
						ushort v = HandModel_depthMap.at<ushort>(y, x);
						if (v != 0) {
							HandModel_depthMap.at<ushort>(y, x) = min(v, (ushort)depth);
						}
						else {
							HandModel_depthMap.at<ushort>(y, x) = (ushort)depth;
						}
					}
				}
			}
		}
	}
}

Eigen::Vector3f HandModel::ComputePalmCenterPosition(Eigen::RowVector3f& palm_center)
{
	Eigen::RowVector3f palmCenter = Eigen::RowVector3f::Zero();

	palmCenter(0) = 0.33*this->J_Final(0, 0) + 0.33*this->J_Final(1, 0) + 0.33*this->J_Final(7, 0) - this->J_Final(0, 0);
	palmCenter(1) = 0.33*this->J_Final(0, 1) + 0.33*this->J_Final(1, 1) + 0.33*this->J_Final(7, 1) - this->J_Final(0, 1);
	palmCenter(2) = 0.33*this->J_Final(0, 2) + 0.33*this->J_Final(1, 2) + 0.33*this->J_Final(7, 2) - this->J_Final(0, 2);

	if (isnan(palmCenter(0)) || isnan(palmCenter(1)) || isnan(palmCenter(2))
		|| isinf(palmCenter(0)) || isinf(palmCenter(1)) || isinf(palmCenter(2)))
	{
		palmCenter = Eigen::RowVector3f::Zero();
	}

	return (palm_center - this->J_shaped.row(0) - palmCenter).transpose();
}