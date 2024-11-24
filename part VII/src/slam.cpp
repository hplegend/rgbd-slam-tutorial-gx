/*************************************************************************
	> File Name: rgbd-slam-tutorial-gx/part V/src/visualOdometry.cpp
	> Author: xiang gao
	> Mail: gaoxiang12@mails.tsinghua.edu.cn
	> Created Time: 2015年08月15日 星期六 15时35分42秒
    * add g2o slam end to visual odometry
    * add keyframe and simple loop closure
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;
#include <Eigen/Core>
#include <Eigen/Geometry> // For Eigen::Affine3d
#include "slamBase.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <dirent.h>
#include <sys/types.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm> // std::sort

// 把g2o的定义放到前面
typedef g2o::BlockSolver_6_3 SlamBlockSolver;
typedef g2o::LinearSolverEigen< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;

vector<string> listPathFiles(string path);

// 给定index，读取一帧数据
FRAME readFrame( int index, ParameterReader& pd );

FRAME readMyFrame(string rgb, string depth, int frameId, ParameterReader &pd) ;
void drawCamera(pcl::visualization::PCLVisualizer::Ptr viewer, const  Eigen::Matrix4d& pose, int id);
// 估计一个运动的大小
double normofTransform( cv::Mat rvec, cv::Mat tvec );

// 检测两个帧，结果定义
enum CHECK_RESULT {NOT_MATCHED=0, TOO_FAR_AWAY, TOO_CLOSE, KEYFRAME};
// 函数声明
CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops=false );
// 检测近距离的回环
void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );
// 随机检测回环
void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti );

int main( int argc, char** argv )
{
    // 前面部分和vo是一样的
    ParameterReader pd;
    int startIndex  =   atoi( pd.getData( "start_index" ).c_str() );
    int endIndex    =   atoi( pd.getData( "end_index"   ).c_str() );
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");
    vector<string> depthFilesPath = listPathFiles(depthDir);
    vector<string> rgbFilesPath = listPathFiles(rgbDir);

    // 所有的关键帧都放在了这里
    vector< FRAME > keyframes;
    // initialize
    cout<<"Initializing ..."<<endl;
    int currIndex = startIndex; // 当前索引为currIndex
    FRAME currFrame = readMyFrame( rgbFilesPath[currIndex],depthFilesPath[currIndex], currIndex,pd ); // 上一帧数据

    string detector = pd.getData( "detector" );
    string descriptor = pd.getData( "descriptor" );
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    computeKeyPointsAndDesp( currFrame, detector, descriptor );
    PointCloud::Ptr cloud = image2PointCloud( currFrame.rgb, currFrame.depth, camera );

    /******************************* 
    // 新增:有关g2o的初始化
    *******************************/
    // 初始化求解器
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );

    g2o::SparseOptimizer globalOptimizer;  // 最后用的就是这个东东
    globalOptimizer.setAlgorithm( solver );
    // 不要输出调试信息
    globalOptimizer.setVerbose( false );

    // 向globalOptimizer增加第一个顶点
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId( currIndex );
    v->setEstimate( Eigen::Isometry3d::Identity() ); //估计为单位矩阵
    v->setFixed( true ); //第一个顶点固定，不用优化
    globalOptimizer.addVertex( v );
  //  pcl::visualization::CloudViewer viewer("viewer");

    // 是否显示点云
    bool visualize = pd.getData("visualize_pointcloud")==string("yes");

    currFrame.rvec = cv::Mat::zeros(3, 3, CV_64F);
    currFrame.tvec = cv::Mat::zeros(1, 3, CV_64F);
    keyframes.push_back( currFrame );


    double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    bool check_loop_closure = pd.getData("check_loop_closure")==string("yes");
    int lostFrameCnt = 0;
    for ( currIndex=startIndex+1; currIndex<endIndex; currIndex++ )
    {
        cout<<"Reading files "<<currIndex<<endl;
        FRAME currFrame = readMyFrame( rgbFilesPath[currIndex],depthFilesPath[currIndex],currIndex,pd ); // 读取currFrame
        computeKeyPointsAndDesp( currFrame, detector, descriptor ); //提取特征
        CHECK_RESULT result = checkKeyframes( keyframes.back(), currFrame, globalOptimizer ); //匹配该帧与keyframes里最后一帧
        Eigen::Isometry3d T = cvMat2Eigen( currFrame.rvec, currFrame.tvec );
        cout<<"T="<<T.matrix()<<endl;

        switch (result) // 根据匹配结果不同采取不同策略
        {
        case NOT_MATCHED:
            //没匹配上，直接跳过
            cout<<RED"Not enough inliers."<<endl;
            lostFrameCnt = lostFrameCnt + 1;
//            if(lostFrameCnt >=1) {// 连续两帧没有跟上
//                // 计算方向向量，计算速度，插入一帧，或者从新开始一帧
//                // 目的是保证地图不断。
//                pcl::visualization::PCLVisualizer::Ptr camViewer(new pcl::visualization::PCLVisualizer("SLAM Camera Visualization"));
//                camViewer->setBackgroundColor(0, 0, 0);
//                camViewer->addCoordinateSystem(1.0); // 参数为坐标轴长度
//
//                camViewer->removeAllPointClouds();
//                camViewer->removeAllShapes();
//
//                Eigen::Matrix4d matrix4D = Eigen::Matrix4d::Identity();
//                pcl::PointXYZ last_O_pos;
//                for (size_t i = 0; i < keyframes.size(); ++i) {
//                    pcl::PointXYZ O_pose;
//                    Eigen::Isometry3d T = cvMat2Eigen( keyframes[i].rvec, keyframes[i].tvec );
//                    //每帧位姿的原点坐标只由变换矩阵中的平移向量得到
//                    O_pose.x = T.translation()[0];
//                    O_pose.y = T.translation()[1];
//                    O_pose.z = T.translation()[2];
//                    if (i > 0)
//                        camViewer->addLine(last_O_pos, O_pose, 255, 255, 255, "trac_" + std::to_string(i));
//
//                    pcl::PointXYZ X;
//                    Eigen::Vector3d Xw =T * (0.1 * Eigen::Vector3d(1, 0, 0));
//                    X.x = Xw[0];
//                    X.y = Xw[1];
//                    X.z = Xw[2];
//                    camViewer->addLine(O_pose, X, 255, 0, 0, "X_" + std::to_string(i));
//
//                    pcl::PointXYZ Y;
//                    Eigen::Vector3d Yw =T * (0.1 * Eigen::Vector3d(0, 1, 0));
//                    Y.x = Yw[0];
//                    Y.y = Yw[1];
//                    Y.z = Yw[2];
//                    camViewer->addLine(O_pose, Y, 0, 255, 0, "Y_" + std::to_string(i));
//
//                    pcl::PointXYZ Z;
//                    Eigen::Vector3d Zw =T * (0.1 * Eigen::Vector3d(0, 0, 1));
//                    Z.x = Zw[0];
//                    Z.y = Zw[1];
//                    Z.z = Zw[2];
//                    camViewer->addLine(O_pose, Z, 0, 0, 255, "Z_" + std::to_string(i));
//
//                    last_O_pos = O_pose;
//
//                //    drawCamera(camViewer, matrix4D, i); // 绘制相机
//                }
//
//                while (!camViewer->wasStopped()) {
//                    camViewer->spinOnce(100);
//                }
//
//            }


            break;
        case TOO_FAR_AWAY:
            // 太近了，也直接跳
            cout<<RED"Too far away, may be an error."<<endl;
            break;
        case TOO_CLOSE:
            // 太远了，可能出错了
            cout<<RESET"Too close, not a keyframe"<<endl;
         //   keyframes.push_back( currFrame );
            break;
        case KEYFRAME:
            cout<<GREEN"This is a new keyframe"<<endl;
            lostFrameCnt = 0;
            // 不远不近，刚好
            /**
             * This is important!!
             * This is important!!
             * This is important!!
             * (very important so I've said three times!)
             */
            // 检测回环
            if (check_loop_closure)
            {
                checkNearbyLoops( keyframes, currFrame, globalOptimizer );
                checkRandomLoops( keyframes, currFrame, globalOptimizer );
            }
            keyframes.push_back( currFrame );


//                cloud = joinPointCloud( cloud, currFrame, T, camera );
//
//                if ( visualize == true )
//                    viewer.showCloud( cloud );

            break;
        default:
            break;
        }

    }

    // 优化
    cout<<RESET"optimizing pose graph, vertices: "<<globalOptimizer.vertices().size()<<endl;
    globalOptimizer.save("../result_before.g2o");
    globalOptimizer.initializeOptimization();
    globalOptimizer.optimize( 100 ); //可以指定优化步数
    globalOptimizer.save( "../result_after.g2o" );
    cout<<"Optimization done."<<endl;

    // 拼接点云地图
    cout<<"saving the point cloud map..."<<endl;
    PointCloud::Ptr output ( new PointCloud() ); //全局地图
    PointCloud::Ptr tmp ( new PointCloud() );

    pcl::VoxelGrid<PointT> voxel; // 网格滤波器，调整地图分辨率
    pcl::PassThrough<PointT> pass; // z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉
    pass.setFilterFieldName("z");
    pass.setFilterLimits( 0.0, 4.0 ); //4m以上就不要了

    double gridsize = atof( pd.getData( "voxel_grid" ).c_str() ); //分辨图可以在parameters.txt里调
    voxel.setLeafSize( gridsize, gridsize, gridsize );

//    for (size_t i=0; i<keyframes.size(); i++)
//    {
//        // 从g2o里取出一帧
//        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID ));
//        Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿
//        PointCloud::Ptr newCloud = image2PointCloud( keyframes[i].rgb, keyframes[i].depth, camera ); //转成点云
//        // 以下是滤波
////        voxel.setInputCloud( newCloud );
////        voxel.filter( *tmp );
////        pass.setInputCloud( tmp );
////        pass.filter( *newCloud );
//        // 把点云变换后加入全局地图中
//        pcl::transformPointCloud( *newCloud, *tmp, pose.matrix() );
//        *output += *tmp;
//        tmp->clear();
//        newCloud->clear();
//    }

    // 计算方向向量，计算速度，插入一帧，或者从新开始一帧
    // 目的是保证地图不断。
    pcl::visualization::PCLVisualizer::Ptr camViewer(
            new pcl::visualization::PCLVisualizer("SLAM Camera Visualization"));
    camViewer->setBackgroundColor(0, 0, 0);
    camViewer->addCoordinateSystem(0.1); // 参数为坐标轴长度

    camViewer->removeAllPointClouds();
    camViewer->removeAllShapes();



    g2o::VertexSE3* vertexR2 = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[keyframes.size()-2].frameID ));
    Eigen::Isometry3d poseR2 = vertexR2->estimate(); //该帧优化后的位姿

    g2o::VertexSE3* vertexR1 = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[keyframes.size()-1].frameID ));
    Eigen::Isometry3d poseR1 = vertexR1->estimate(); //该帧优化后的位姿

    Eigen::Vector3d dirR2(poseR2.translation()[0], poseR2.translation()[1], poseR2.translation()[2]);
    Eigen::Vector3d dirR1(poseR1.translation()[0], poseR1.translation()[1], poseR1.translation()[2]);
    Eigen::Vector3d add = dirR1 + dirR2;
    double magnitude = add.norm(); // 计算模
    Eigen::Vector3d velocity = (add / magnitude) * 0.01;
    // 计算一个方向向量，和速度
    // 添加100帧看看效果
    Eigen::Matrix4d matrix4D = Eigen::Matrix4d::Identity();
    pcl::PointXYZ last_O_pos;
    for (size_t i = 0; i < keyframes.size(); ++i) {
        // 从g2o里取出一帧
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[i].frameID ));
        Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿

        pcl::PointXYZ O_pose;
        //每帧位姿的原点坐标只由变换矩阵中的平移向量得到
        O_pose.x = pose.translation()[0];
        O_pose.y = pose.translation()[1];
        O_pose.z = pose.translation()[2];
        cout<<O_pose.x<<","<<O_pose.y<<","<<O_pose.z<<endl;
        if (i > 0)
            camViewer->addLine(last_O_pos, O_pose, 255, 255, 255, "trac_" + std::to_string(i));

        pcl::PointXYZ X;
        Eigen::Vector3d Xw = pose * (0.1 * Eigen::Vector3d(1, 0, 0));
        X.x = Xw[0];
        X.y = Xw[1];
        X.z = Xw[2];
        camViewer->addLine(O_pose, X, 255, 0, 0, "X_" + std::to_string(i));

        pcl::PointXYZ Y;
        Eigen::Vector3d Yw = pose * (0.1 * Eigen::Vector3d(0, 1, 0));
        Y.x = Yw[0];
        Y.y = Yw[1];
        Y.z = Yw[2];
        camViewer->addLine(O_pose, Y, 0, 255, 0, "Y_" + std::to_string(i));

        pcl::PointXYZ Z;
        Eigen::Vector3d Zw = pose * (0.1 * Eigen::Vector3d(0, 0, 1));
        Z.x = Zw[0];
        Z.y = Zw[1];
        Z.z = Zw[2];
        camViewer->addLine(O_pose, Z, 0, 0, 255, "Z_" + std::to_string(i));

        last_O_pos = O_pose;

    }

    for (size_t i = 200; i < 400; ++i) {

        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(globalOptimizer.vertex( keyframes[keyframes.size()-1].frameID ));
        Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿
        pose.translation()[1] = last_O_pos.y + velocity[1];

        pcl::PointXYZ O_pose;
        //每帧位姿的原点坐标只由变换矩阵中的平移向量得到
        O_pose.x = pose.translation()[0];
        O_pose.y = pose.translation()[1];
        O_pose.z = pose.translation()[2];

        cout<< "aaa:"<<O_pose.x<<","<<O_pose.y<<","<<O_pose.z<<endl;

        camViewer->addLine(last_O_pos, O_pose, 255, 255, 255, "trac_" + std::to_string(i));

        pcl::PointXYZ X;
        Eigen::Vector3d Xw = pose * (0.1 * Eigen::Vector3d(1, 0, 0));
        X.x = Xw[0];
        X.y = Xw[1];
        X.z = Xw[2];
        camViewer->addLine(O_pose, X, 255, 0, 0, "X_" + std::to_string(i));

        pcl::PointXYZ Y;
        Eigen::Vector3d Yw = pose * (0.1 * Eigen::Vector3d(0, 1, 0));
        Y.x = Yw[0];
        Y.y = Yw[1];
        Y.z = Yw[2];
        camViewer->addLine(O_pose, Y, 0, 255, 0, "Y_" + std::to_string(i));

        pcl::PointXYZ Z;
        Eigen::Vector3d Zw = pose * (0.1 * Eigen::Vector3d(0, 0, 1));
        Z.x = Zw[0];
        Z.y = Zw[1];
        Z.z = Zw[2];
        camViewer->addLine(O_pose, Z, 0, 0, 255, "Z_" + std::to_string(i));

        last_O_pos = O_pose;
    }



    while (!camViewer->wasStopped()) {
        camViewer->spinOnce(100);
    }


//    voxel.setInputCloud( output );
//    voxel.filter( *tmp );
    //存储
    pcl::io::savePCDFile( "../result.pcd", *tmp );

    cout<<"Final map is saved."<<endl;
    return 0;
}

FRAME readFrame( int index, ParameterReader& pd )
{
    FRAME f;
    string rgbDir   =   pd.getData("rgb_dir");
    string depthDir =   pd.getData("depth_dir");

    string rgbExt   =   pd.getData("rgb_extension");
    string depthExt =   pd.getData("depth_extension");

    stringstream ss;
    ss<<rgbDir<<index<<rgbExt;
    string filename;
    ss>>filename;
    f.rgb = cv::imread( filename );

    ss.clear();
    filename.clear();
    ss<<depthDir<<index<<depthExt;
    ss>>filename;

    f.depth = cv::imread( filename, -1 );
    f.frameID = index;
    return f;
}

double normofTransform( cv::Mat rvec, cv::Mat tvec )
{
    return fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
}

CHECK_RESULT checkKeyframes( FRAME& f1, FRAME& f2, g2o::SparseOptimizer& opti, bool is_loops)
{
    static ParameterReader pd;
    static int min_inliers = atoi( pd.getData("min_inliers").c_str() );
    static double max_norm = atof( pd.getData("max_norm").c_str() );
    static double keyframe_threshold = atof( pd.getData("keyframe_threshold").c_str() );
    static double max_norm_lp = atof( pd.getData("max_norm_lp").c_str() );
    static CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    // 比较f1 和 f2
    RESULT_OF_PNP result = estimateMotion( f1, f2, camera );
    f2.tvec = result.tvec;
    f2.rvec = result.rvec;
    if ( result.inliers < min_inliers ) //inliers不够，放弃该帧
        return NOT_MATCHED;
    // 计算运动范围是否太大
    double norm = normofTransform(result.rvec, result.tvec);
    if ( is_loops == false )
    {
        if ( norm >= max_norm )
            return TOO_FAR_AWAY;   // too far away, may be error
    }
    else
    {
        if ( norm >= max_norm_lp)
            return TOO_FAR_AWAY;
    }

    if ( norm <= keyframe_threshold )
        return TOO_CLOSE;   // too adjacent frame
    // 向g2o中增加这个顶点与上一帧联系的边
    // 顶点部分
    // 顶点只需设定id即可
    if (is_loops == false)
    {
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId( f2.frameID );
        v->setEstimate( Eigen::Isometry3d::Identity() );
        opti.addVertex(v);
    }
    // 边部分
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
    // 连接此边的两个顶点id
    edge->setVertex( 0, opti.vertex(f1.frameID ));
    edge->setVertex( 1, opti.vertex(f2.frameID ));
    edge->setRobustKernel( new g2o::RobustKernelHuber() );
    // 信息矩阵
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;
    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation( information );
    // 边的估计即是pnp求解之结果
    Eigen::Isometry3d T = cvMat2Eigen( result.rvec, result.tvec );
    // edge->setMeasurement( T );
    edge->setMeasurement( T.inverse() );
    // 将此边加入图中
    opti.addEdge(edge);
    return KEYFRAME;
}

void checkNearbyLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti )
{
    static ParameterReader pd;
    static int nearby_loops = atoi( pd.getData("nearby_loops").c_str() );

    // 就是把currFrame和 frames里末尾几个测一遍
    if ( frames.size() <= nearby_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, opti, true );
        }
    }
    else
    {
        // check the nearest ones
        for (size_t i = frames.size()-nearby_loops; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, opti, true );
        }
    }
}

void checkRandomLoops( vector<FRAME>& frames, FRAME& currFrame, g2o::SparseOptimizer& opti )
{
    static ParameterReader pd;
    static int random_loops = atoi( pd.getData("random_loops").c_str() );
    srand( (unsigned int) time(NULL) );
    // 随机取一些帧进行检测

    if ( frames.size() <= random_loops )
    {
        // no enough keyframes, check everyone
        for (size_t i=0; i<frames.size(); i++)
        {
            checkKeyframes( frames[i], currFrame, opti, true );
        }
    }
    else
    {
        // randomly check loops
        for (int i=0; i<random_loops; i++)
        {
            int index = rand()%frames.size();
            checkKeyframes( frames[index], currFrame, opti, true );
        }
    }
}


FRAME readMyFrame(string rgb, string depth, int frameId, ParameterReader &pd) {
    FRAME f;
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");

    string rgbExt = pd.getData("rgb_extension");
    string depthExt = pd.getData("depth_extension");

    f.rgb = cv::imread(rgb);
    f.depth = cv::imread(depth, -1);
    f.frameID = frameId;
    return f;
}


vector<string> listPathFiles(string path) {
    char namebuf[255] = {0};

    vector<string> ret = {};

    DIR *dirp = NULL;
    struct dirent *dir_entry = NULL;
    char file_path[100] = {0};
    if ((dirp = opendir(path.c_str())) == NULL) {
        printf("Opendir %s fail!\n", file_path);
        return ret;
    }
    while ((dir_entry = readdir(dirp)) != NULL) {
        if (strcmp(dir_entry->d_name, ".") == 0 || strcmp(dir_entry->d_name, "..") == 0) {
            continue;
        }
        sprintf(namebuf, "%s%s", path.c_str(), dir_entry->d_name);
        stringstream ss;
        ss << namebuf;
        ret.push_back(ss.str());
    }
    std::sort(ret.begin(), ret.end());
    return ret;
}


// 绘制相机模型（用箭头表示）
void drawCamera(pcl::visualization::PCLVisualizer::Ptr viewer,  const Eigen::Matrix4d & pose, int id) {
    Eigen::Vector3d origin(pose(0, 3), pose(1, 3), pose(2, 3)); // 相机中心
    Eigen::Vector3d forward(pose(0, 2), pose(1, 2), pose(2, 2)); // 相机方向

    // 生成箭头表示相机方向
    pcl::PointXYZ start(origin[0], origin[1], origin[2]);
    pcl::PointXYZ end(origin[0] + 0.1 * forward[0],
                      origin[1] + 0.1 * forward[1],
                      origin[2] + 0.1 * forward[2]);

    std::string arrow_name = "camera_arrow_" + std::to_string(id);
    viewer->addSphere(start,0.05,arrow_name);
   // viewer->addArrow(end, start, 1.0, 0.0, 0.0, false, arrow_name); // 红色箭头
}