#ifndef REMESH_TYPE
#define REMESH_TYPE
#include <Eigen/Core>
#include <igl/opengl/glfw/Viewer.h>

struct Buffer
{
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B;
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A;

    Buffer(int width = 512, int height = 512);

    void operator*=(const Buffer& b);
};
#endif
