//
// Created by hplegend on 24-11-23.
//

#ifndef SLAM_DRAWSLAM_H
#define SLAM_DRAWSLAM_H
#include <GL/glut.h>
#include <Eigen/Core>
#include <Eigen/Geometry> // For Eigen::Affine3d
#include <vector>
class drawSlam {

public:
    void initCameraPoses();
    void drawCamera();
    void drawTrajectory();
    void display();
    void idle();
    // 相机位姿存储：每个位姿是一个 4x4 矩阵
    std::vector<Eigen::Matrix4d> camera_poses;
};


#endif //SLAM_DRAWSLAM_H
