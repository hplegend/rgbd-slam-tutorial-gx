//
// Created by hplegend on 24-11-23.
//

#include "drawSlam.h"
void drawSlam::initCameraPoses() {
    for (int i = 0; i < 50; ++i) {
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose(0, 3) = i * 0.1; // x 方向平移
        pose(1, 3) = 0.5 * std::sin(i * 0.1); // y 方向变化
        camera_poses.push_back(pose);
    }
}
void drawSlam::drawCamera() {
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(0.2f, 0.1f, -0.2f);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(0.2f, -0.1f, -0.2f);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(-0.2f, -0.1f, -0.2f);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(-0.2f, 0.1f, -0.2f);
    glVertex3f(0.2f, 0.1f, -0.2f); glVertex3f(0.2f, -0.1f, -0.2f);
    glVertex3f(-0.2f, 0.1f, -0.2f); glVertex3f(-0.2f, -0.1f, -0.2f);
    glVertex3f(0.2f, 0.1f, -0.2f); glVertex3f(-0.2f, 0.1f, -0.2f);
    glVertex3f(0.2f, -0.1f, -0.2f); glVertex3f(-0.2f, -0.1f, -0.2f);
    glEnd();
}


// 绘制轨迹
void drawSlam::drawTrajectory() {
    glBegin(GL_LINE_STRIP);
    glColor3f(1.0f, 0.0f, 0.0f); // 红色轨迹
    for (const auto& pose : camera_poses) {
        glVertex3f(pose(0, 3), pose(1, 3), pose(2, 3)); // 提取平移向量
    }
    glEnd();
}

// OpenGL 显示回调
void drawSlam::display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    gluLookAt(3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    // 绘制每个相机的位置
    for (const auto& pose : camera_poses) {
        glPushMatrix();
        glMultMatrixd(pose.data()); // 应用相机位姿变换
        glColor3f(0.0f, 1.0f, 0.0f); // 绿色相机
        drawCamera();
        glPopMatrix();
    }

    // 绘制轨迹
    drawTrajectory();

    glutSwapBuffers();
}
