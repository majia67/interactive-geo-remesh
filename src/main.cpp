#include <iostream>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/jet.h>
#include <igl/principal_curvature.h>
#include <igl/gaussian_curvature.h>
#include <igl/triangle/triangulate.h>
#include <Eigen/Core>

#include "varcoeffED.h"

#define BLACK		0
#define WHITE		255

using namespace Eigen;
using namespace std;

typedef Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> MatrixXuc;

MatrixXd V(0, 3);                //vertex array, #V x3
MatrixXi F(0, 3);                //face array, #F x3
MatrixXd V_uv(0, 2);             //vertex array in the UV plane, #V x2

VectorXd area_map;               //area map, #F x1
VectorXd mean_curv_map;
VectorXd gaus_curv_map;          //gaussian curvature map, #V x1
MatrixXi control_map;            //control map, pixel width x height
MatrixXi sampling_data;          //sampling result, pixel width x height

struct Option
{
    // Geometry maps
    bool is_inverse_mode;

    // Control map
    bool use_area_map;
    bool use_mean_curv_map;
    bool use_gaus_curv_map;
    float scaling_factor;
} options;

void reset_mesh(igl::opengl::glfw::Viewer &viewer);
void map_vertices_to_rectangle(const Eigen::MatrixXd& V, const Eigen::VectorXi& bnd, Eigen::MatrixXd& UV);
void harmonic_parameterization();
void calc_area_map();
void calc_mean_curvature_map();
void calc_gaussian_curvature_map();
void calc_control_map(igl::opengl::glfw::Viewer &viewer);
void render_map(igl::opengl::glfw::Viewer &viewer, VectorXd &map);
void render_pixel_img(igl::opengl::glfw::Viewer &viewer, MatrixXi &img);
void sampling();
void get_uv_coord_from_pixel_img(const MatrixXi &img, MatrixXd &UV, MatrixXi &E);
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

    // Initialize options
    options.is_inverse_mode = true;
    options.use_area_map = true;
    options.use_mean_curv_map = true;
    options.use_gaus_curv_map = false;
    options.scaling_factor = 1.0;

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

        ImGui::Checkbox("Inverse Mode", &options.is_inverse_mode);

        if (ImGui::CollapsingHeader("Parameterization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Harmonic"))
            {
                harmonic_parameterization();

                viewer.data().clear();
                viewer.data().set_mesh(V_uv, F);
                viewer.data().set_uv(V_uv);
                viewer.core.align_camera_center(V_uv, F);
                viewer.data().show_lines = true;
            }
        }

        if (ImGui::CollapsingHeader("Geometry Maps", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Area Map"))
            {
                calc_area_map();
                render_map(viewer, area_map);
            }

            if (ImGui::Button("Mean Curvature Map"))
            {
                calc_mean_curvature_map();
                render_map(viewer, mean_curv_map);
            }

            if (ImGui::Button("Gaussian Curvature Map"))
            {
                calc_gaussian_curvature_map();
                render_map(viewer, gaus_curv_map);
            }
        }

        if (ImGui::CollapsingHeader("Control Map", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Use area map", &options.use_area_map);
            ImGui::Checkbox("Use mean curvature map", &options.use_mean_curv_map);
            ImGui::Checkbox("Use Gaussian Curvature map", &options.use_gaus_curv_map);
            ImGui::InputFloat("scaling factor", &options.scaling_factor);

            if (ImGui::Button("Generate Control Map"))
            {
                calc_control_map(viewer);
                render_pixel_img(viewer, control_map);
            }
        }

        if (ImGui::CollapsingHeader("Sampling", ImGuiTreeNodeFlags_DefaultOpen))
        {
            //ImGui::InputInt("samples", &num_of_samples);

            if (ImGui::Button("Perform Sampling"))
            {
                cout << control_map.rows() << " x " << control_map.cols() << endl;

                //int count = (control_map.array() < 127.5).count();
                //cout << count << endl;

                //control_map *= (double)num_of_samples / (control_map.rows() * control_map.cols());
                //// Normalize the pixel intensity
                //int max = control_map.maxCoeff();
                //for (int i = 0; i < control_map.size(); i++)
                //{
                //    control_map(i) = (double)control_map(i) / max * 255;
                //}
                //cout << (control_map.array() < 127.5).count() << endl;

                sampling();
                render_pixel_img(viewer, sampling_data);
            }
        }

        if (ImGui::Button("Triangulate"))
        {
            MatrixXd UV, H, V2;
            MatrixXi E, F2;
            get_uv_coord_from_pixel_img(sampling_data, UV, E);

            igl::triangle::triangulate(UV, E, H, "a0.005q", V2, F2);

            viewer.data().clear();
            viewer.data().set_mesh(V2, F2);
            viewer.core.align_camera_center(V2, F2);
            viewer.data().show_lines = true;
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
    viewer.data().show_lines = true;
}

void harmonic_parameterization()
{
    // Find the open boundary
    VectorXi bnd;
    igl::boundary_loop(F, bnd);

    // Map the boundary to a circle, preserving edge proportions
    MatrixXd bnd_uv;
    map_vertices_to_rectangle(V, bnd, bnd_uv);

    // Harmonic parametrization for the internal vertices
    igl::harmonic(V, F, bnd, bnd_uv, 1, V_uv);
}

void map_vertices_to_rectangle(
    const Eigen::MatrixXd& V,
    const Eigen::VectorXi& bnd,
    Eigen::MatrixXd& UV)
{
    std::vector<double> len(bnd.size());
    len[0] = 0.0;

    for (int i = 1; i < bnd.size(); i++)
    {
        len[i] = len[i - 1] + (V.row(bnd[i - 1]) - V.row(bnd[i])).norm();
    }
    double total_len = len[len.size() - 1] + (V.row(bnd[0]) - V.row(bnd[bnd.size() - 1])).norm();

    UV.resize(bnd.size(), 2);
    for (int i = 0; i < bnd.size(); i++)
    {
        double frac = len[i] * 4.0 / total_len;
        if (frac <= 1.)
            UV.row(i) << frac, 0.0;
        else if (frac <= 2.)
            UV.row(i) << 1.0, frac - 1.0;
        else if (frac <= 3.)
            UV.row(i) << 3.0 - frac, 1.0;
        else
            UV.row(i) << 0.0, 4.0 - frac;
    }
}

void calc_area_map()
{
    ArrayXd dblA3D;
    ArrayXd dblA2D;

    igl::doublearea(V, F, dblA3D);
    igl::doublearea(V_uv, F, dblA2D);

    area_map.resize(F.rows());
    area_map << dblA3D / dblA2D;
}

void calc_mean_curvature_map()
{
    MatrixXd PD1, PD2;
    VectorXd PV1, PV2;
    igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);
    
    mean_curv_map.resize(V.rows());
    mean_curv_map = (PV1.array() + PV2.array()) / 2;
}

void calc_gaussian_curvature_map()
{
    // Calculate per-vertex discrete gaussian curvature
    igl::gaussian_curvature(V, F, gaus_curv_map);
}

void calc_control_map(igl::opengl::glfw::Viewer &viewer)
{
    const int width = (int)viewer.core.viewport(2);
    const int height = (int)viewer.core.viewport(3);

    MatrixXuc R(width, height);
    MatrixXuc G(width, height);
    MatrixXuc B(width, height);
    MatrixXuc A(width, height);
    MatrixXd temp = MatrixXd::Ones(width, height);
    
    if (options.use_area_map)
    {
        render_map(viewer, area_map);
        viewer.core.draw_buffer(viewer.data(), false, R, G, B, A);
        temp = temp.array() * R.cast<double>().array();
    }

    if (options.use_mean_curv_map)
    {
        render_map(viewer, mean_curv_map);
        viewer.core.draw_buffer(viewer.data(), false, R, G, B, A);
        temp = temp.array() * R.cast<double>().array();
    }

    if (options.use_gaus_curv_map)
    {
        render_map(viewer, gaus_curv_map);
        viewer.core.draw_buffer(viewer.data(), false, R, G, B, A);
        temp = temp.array() * R.cast<double>().array();
    }

    // Normalize the pixel intensity
    temp *= 255.0 / temp.maxCoeff();

    // Linear scale the intensity
    temp *= options.scaling_factor;
    for (int i = 0; i < temp.size(); i++)
    {
        temp(i) -= 255.0 * (int)(temp(i) / 255.0);
    }

    // Crop the boarder and only keep the map inside
    int top = 0, bottom = A.rows() - 1, left = 0, right = A.cols() - 1;
    
    while (A.row(top).maxCoeff() == A.row(top).minCoeff()) top++;
    while (A.row(bottom).maxCoeff() == A.row(bottom).minCoeff()) bottom--;
    while (A.col(left).maxCoeff() == A.col(left).minCoeff()) left++;
    while (A.col(right).maxCoeff() == A.col(right).minCoeff()) right--;

    int rows = bottom - top + 1, cols = right - left + 1;
    control_map.resize(rows, cols);
    control_map = temp.block(top, left, rows, cols).cast<int>();
}

void render_map(igl::opengl::glfw::Viewer &viewer, VectorXd &map)
{
    MatrixXd color;
    grayscale_jet(map, color);

    viewer.data().clear();
    viewer.data().set_mesh(V_uv, F);
    viewer.core.align_camera_center(V_uv);
    viewer.data().set_colors(color);
    viewer.data().show_texture = false;
    viewer.data().show_lines = false;
    viewer.data().V_material_specular.setZero();
    viewer.data().F_material_specular.setZero();
}

void render_pixel_img(igl::opengl::glfw::Viewer &viewer, MatrixXi &img)
{
    // Replace the mesh with a triangulated square
    MatrixXd V(4, 3);
    V <<
        0, 1, 0,
        0, 0, 0,
        1, 0, 0,
        1, 1, 0;
    MatrixXi F(2, 3);
    F <<
        0, 1, 2,
        2, 3, 0;
    MatrixXd UV(4, 2);
    UV <<
        0, 1,
        0, 0,
        1, 0,
        1, 1;

    MatrixXuc K = img.cast<unsigned char>();

    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_uv(UV);
    viewer.core.align_camera_center(V);
    viewer.data().show_texture = true;
    viewer.data().show_lines = false;
    viewer.data().set_texture(K, K, K);

    MatrixXd color = MatrixXd::Ones(V.rows(), V.cols());
    viewer.data().set_colors(color);

    viewer.data().V_material_specular.setZero();
    viewer.data().F_material_specular.setZero();
}

void sampling()
{
    sampling_data.resize(control_map.rows(), control_map.cols());
    error_diffusion(control_map, sampling_data);
}

void get_uv_coord_from_pixel_img(const MatrixXi &img, MatrixXd &UV, MatrixXi &E)
{
    int num, count;
    num = (img.array() == BLACK).count();
    count = 0;
    UV.resize(num, 2);

    // Adding boundary points in counter-clockwise order
    for (int c = 0; c < img.cols(); c++)
    {
        if (img(img.rows() - 1, c) == BLACK)
        {
            UV.row(count) << (double)c / (img.cols() - 1), 0.0;
            count++;
        }
    }

    for (int r = img.rows() - 2; r >= 0; r--)
    {
        if (img(r, img.cols() - 1) == BLACK)
        {
            UV.row(count) << 1.0, 1.0 - (double)r / (img.rows() - 1);
            count++;
        }
    }

    for (int c = img.cols() - 2; c >= 0; c--)
    {
        if (img(0, c) == BLACK)
        {
            UV.row(count) << (double)c / (img.cols() - 1), 1.0;
            count++;
        }
    }

    for (int r = 1; r < img.rows(); r++)
    {
        if (img(r, 0) == BLACK)
        {
            UV.row(count) << 0.0, 1.0 - (double)r / (img.rows() - 1);
            count++;
        }
    }

    // Generate the boundary edges
    E.resize(count, 2);
    for (int i = 0; i < count - 1; i++)
    {
        E.row(i) << i, i + 1;
    }
    E.row(count - 1) << count - 1, 0;

    // Process the interior points
    for (int r = 1; r < img.rows() - 1; r++)
    {
        for (int c = 1; c < img.cols() - 1; c++)
        {
            if (img(r, c) == BLACK)
            {
                UV.row(count) << (double)c / img.cols(), (double)r / img.rows();
                count++;
            }
        }
    }
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
        if (options.is_inverse_mode)
        {
            norm = 1 - norm;
        }
        color(i, 0) = norm;
        color(i, 1) = norm;
        color(i, 2) = norm;
    }
}
