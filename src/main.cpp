#include <iostream>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/jet.h>
#include <igl/gaussian_curvature.h>
#include <Eigen/Core>

#include "types.hpp"

using namespace Eigen;
using namespace std;

MatrixXd V(0, 3);                //vertex array, #V x3
MatrixXi F(0, 3);                //face array, #F x3
MatrixXd V_uv(0, 2);             //vertex array in the UV plane, #V x2

VectorXd area_map;               //area map, #F x1
VectorXd gaus_curv_map;          //gaussian curvature map, #V x1
Buffer control_map;              //control map, #F x1

int num_of_samples;              //number of samples
bool is_inverse_mode;            //inverse mode control

void reset_mesh(igl::opengl::glfw::Viewer &viewer);
void harmonic_parameterization();
void calc_area_map();
void calc_gaussian_curvature_map();
void calc_control_map(igl::opengl::glfw::Viewer &viewer);
void render_map(igl::opengl::glfw::Viewer &viewer, VectorXd &map);
void sampling();
void grayscale_jet(VectorXd &scalar_map, MatrixXd &color);

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: remesh_bin mesh.off" << std::endl;
        exit(0);
    }

    // Read mesh
    igl::readOFF(argv[1], V, F);
    assert(V.rows() > 0);

    // Initialize variables
    num_of_samples = V.rows();
    is_inverse_mode = true;

    // Plot the mesh
    igl::opengl::glfw::Viewer viewer;
    reset_mesh(viewer);

    // Setup the menu
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);
    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::Button("Reset Mesh"))
        {
            reset_mesh(viewer);
        }

        ImGui::Checkbox("Inverse Mode", &is_inverse_mode);

        if (ImGui::CollapsingHeader("Parameterization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Harmonic"))
            {
                harmonic_parameterization();

                viewer.data().clear();
                viewer.data().set_mesh(V_uv, F);
                viewer.data().set_uv(V_uv);
                viewer.core.align_camera_center(V_uv, F);
            }
        }

        if (ImGui::CollapsingHeader("Geometry Maps", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Area Map"))
            {
                calc_area_map();
                render_map(viewer, area_map);
            }

            if (ImGui::Button("Gaussian Curvature Map"))
            {
                calc_gaussian_curvature_map();
                render_map(viewer, gaus_curv_map);
            }

            if (ImGui::Button("Control Map"))
            {
                calc_control_map(viewer);

                // Replace the mesh with a triangulated square
                MatrixXd V(4, 3);
                V <<
                    -0.5, -0.5, 0,
                    0.5, -0.5, 0,
                    0.5, 0.5, 0,
                    -0.5, 0.5, 0;
                MatrixXi F(2, 3);
                F <<
                    0, 1, 2,
                    2, 3, 0;
                MatrixXd UV(4, 2);
                UV <<
                    0, 0,
                    1, 0,
                    1, 1,
                    0, 1;

                viewer.data().clear();
                viewer.data().set_mesh(V, F);
                viewer.data().set_uv(UV);
                viewer.core.align_camera_center(V);
                viewer.data().show_texture = true;
                viewer.data().set_texture(control_map.R, control_map.G, control_map.B);
            }
        }

        if (ImGui::CollapsingHeader("Sampling", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputInt("samples", &num_of_samples);

            if (ImGui::Button("Perform Sampling"))
            {
                sampling();
            }
        }
    };
    
    viewer.launch();
}

void reset_mesh(igl::opengl::glfw::Viewer &viewer)
{
    viewer.data().clear();

    viewer.data().set_mesh(V, F);
    viewer.core.align_camera_center(V, F);
    viewer.data().show_texture = false;
}

void harmonic_parameterization()
{
    // Find the open boundary
    VectorXi bnd;
    igl::boundary_loop(F, bnd);

    // Map the boundary to a circle, preserving edge proportions
    MatrixXd bnd_uv;
    igl::map_vertices_to_circle(V, bnd, bnd_uv);

    // Harmonic parametrization for the internal vertices
    igl::harmonic(V, F, bnd, bnd_uv, 1, V_uv);
}

void calc_area_map()
{
    ArrayXd dblA3D;
    ArrayXd dblA2D;

    igl::doublearea(V, F, dblA3D);
    igl::doublearea(V_uv, F, dblA2D);

    area_map.resize(F.rows());
    area_map << dblA3D / dblA2D;

    if (is_inverse_mode) {
        area_map = 1 - area_map.array();
    }
}

void calc_gaussian_curvature_map()
{
    // Calculate per-vertex discrete gaussian curvature
    igl::gaussian_curvature(V, F, gaus_curv_map);

    if (is_inverse_mode) {
        gaus_curv_map = 1 - gaus_curv_map.array();
    }
}

void calc_control_map(igl::opengl::glfw::Viewer &viewer)
{
    Buffer temp;

    render_map(viewer, area_map);
    viewer.core.draw_buffer(viewer.data(), false, temp.R, temp.G, temp.B, temp.A);
    control_map = temp;

    render_map(viewer, gaus_curv_map);
    viewer.core.draw_buffer(viewer.data(), false, temp.R, temp.G, temp.B, temp.A);
    control_map *= temp;
}

void render_map(igl::opengl::glfw::Viewer &viewer, VectorXd &map)
{
    MatrixXd color;
    grayscale_jet(map, color);

    viewer.data().clear();
    viewer.data().set_mesh(V_uv, F);
    viewer.data().set_colors(color);
    viewer.data().show_texture = false;
    viewer.data().show_lines = false;
}

void sampling()
{
    
}

void grayscale_jet(VectorXd &scalar_map, MatrixXd &color)
{
    double min_z = scalar_map.minCoeff();
    double max_z = scalar_map.maxCoeff();
    double denom = max_z - min_z;

    color.resize(scalar_map.size(), 3);
    for (int i = 0; i < scalar_map.size(); i++)
    {
        double norm = (scalar_map(i) - min_z) / denom;
        color(i, 0) = norm;
        color(i, 1) = norm;
        color(i, 2) = norm;
    }
}
