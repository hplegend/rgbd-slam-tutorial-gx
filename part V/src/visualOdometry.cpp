/*************************************************************************
	> File Name: rgbd-slam-tutorial-gx/part V/src/visualOdometry.cpp
	> Author: xiang gao
	> Mail: gaoxiang12@mails.tsinghua.edu.cn
	> Created Time: 2015年08月01日 星期六 15时35分42秒
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <sys/types.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm> // std::sort

using namespace std;

#include "slamBase.h"

// 给定index，读取一帧数据
FRAME readFrame(int index, ParameterReader &pd);

FRAME readMyFrame(string rgb, string depth, ParameterReader &pd);


// 度量运动的大小
double normofTransform(cv::Mat rvec, cv::Mat tvec);

vector<string> listPathFiles(string path);


int main(int argc, char **argv) {
    ParameterReader pd;
    int startIndex = atoi(pd.getData("start_index").c_str());
    int endIndex = atoi(pd.getData("end_index").c_str());
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");
    vector<string> depthFilesPath = listPathFiles(depthDir);
    vector<string> rgbFilesPath = listPathFiles(rgbDir);

    // initialize
    cout << "Initializing ..." << endl;
    int currIndex = startIndex; // 当前索引为currIndex
    FRAME lastFrame = readMyFrame(rgbFilesPath[currIndex], depthFilesPath[currIndex], pd); // 上一帧数据
    // 我们总是在比较currFrame和lastFrame
    string detector = pd.getData("detector");
    string descriptor = pd.getData("descriptor");
    CAMERA_INTRINSIC_PARAMETERS camera = getDefaultCamera();
    computeKeyPointsAndDesp(lastFrame, detector, descriptor);
    PointCloud::Ptr cloud = image2PointCloud(lastFrame.rgb, lastFrame.depth, camera);

    pcl::visualization::CloudViewer viewer("viewer");

    // 是否显示点云
    bool visualize = pd.getData("visualize_pointcloud") == string("yes");

    int min_inliers = atoi(pd.getData("min_inliers").c_str());
    double max_norm = atof(pd.getData("max_norm").c_str());

    for (currIndex = startIndex + 1; currIndex < endIndex; currIndex++) {
        cout << "Reading files " << currIndex << endl;
        FRAME currFrame = readMyFrame(rgbFilesPath[currIndex], depthFilesPath[currIndex], pd); // 读取currFrame
        computeKeyPointsAndDesp(currFrame, detector, descriptor);
        // 比较currFrame 和 lastFrame
        RESULT_OF_PNP result = estimateMotion(lastFrame, currFrame, camera);
        if (result.inliers < min_inliers) {//inliers不够，放弃该帧
            cout << "inliers = " << result.inliers << endl;
            continue;
        }
        // 计算运动范围是否太大
        double norm = normofTransform(result.rvec, result.tvec);
        cout << "norm = " << norm << endl;
        if (norm >= max_norm)
            continue;
        Eigen::Isometry3d T = cvMat2Eigen(result.rvec, result.tvec);
        cout << "T=" << T.matrix() << endl;

        cloud = joinPointCloud(cloud, currFrame, T, camera);

        if (visualize)
            viewer.showCloud(cloud);

        lastFrame = currFrame;
    }

    pcl::io::savePCDFile("../data/result.pcd", *cloud);
    return 0;
}

FRAME readFrame(int index, ParameterReader &pd) {
    FRAME f;
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");

    string rgbExt = pd.getData("rgb_extension");
    string depthExt = pd.getData("depth_extension");

    stringstream ss;
    ss << rgbDir << index << rgbExt;
    string filename;
    ss >> filename;
    f.rgb = cv::imread(filename);

    ss.clear();
    filename.clear();
    ss << depthDir << index << depthExt;
    ss >> filename;

    f.depth = cv::imread(filename, -1);
    return f;
}


FRAME readMyFrame(string rgb, string depth, ParameterReader &pd) {
    FRAME f;
    string rgbDir = pd.getData("rgb_dir");
    string depthDir = pd.getData("depth_dir");

    string rgbExt = pd.getData("rgb_extension");
    string depthExt = pd.getData("depth_extension");

    f.rgb = cv::imread(rgb);
    f.depth = cv::imread(depth, -1);
    return f;
}

double normofTransform(cv::Mat rvec, cv::Mat tvec) {
    return fabs(min(cv::norm(rvec), 2 * M_PI - cv::norm(rvec))) + fabs(cv::norm(tvec));
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
