#include <iostream>
#include <map>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/boundary_loop.h>
#include <igl/harmonic.h>
#include <igl/jet.h>
#include <igl/principal_curvature.h>
#include <igl/gaussian_curvature.h>
#include <igl/triangle/triangulate.h>
#include <igl/barycentric_coordinates.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/per_face_normals.h>
#include <igl/slice.h>
#include <igl/colon.h>
#include <Eigen/Core>

#include "varcoeffED.h"

#define BLACK		0
#define WHITE		255

using namespace Eigen;
using namespace std;

typedef Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> MatrixXuc;

struct Color
{
    int R;
    int G;
    int B;

    Color(int r, int g, int b) : R(r), G(g), B(b) {}

    bool operator==(const Color &c) const {
        return R == c.R && G == c.G && B == c.B;
    }

    bool operator<(const Color &c) const {
        return R < c.R || (R == c.R && G < c.G) || (R == c.R && G == c.G && B < c.B);
    }
};

struct Option
{
    // Feature lines
    float dihedral_angle_thresholding;
    bool overlay_feature_lines;

    // Geometry maps
    bool use_inverse_mode;

    // Control map
    bool use_area_map;
    bool use_mean_curv_map;
    bool use_gaus_curv_map;
    float scaling_factor;

    // Sampling
    int num_of_samples;
} options;

MatrixXd V(0, 3);                       //vertex array, #V x3
MatrixXi F(0, 3);                       //face array, #F x3
MatrixXd V_uv(0, 2);                    //vertex array in the UV plane, #V x2
std::map<Color, int> face_idx_map;      //face index map, each face has a unique color
MatrixXd UV_color;                      //color of the faces in UV plane
MatrixXd V2(0, 2);                      //vertex array after triangulation, #V2 x2
MatrixXd V3(0, 3);                      //vertex array after reprojection, #V2 x3
MatrixXi F2(0, 3);                      //face array after triangulation, #F2 x3
MatrixXd bnd_uv;                        //uv coordinates of the boundary points

VectorXd PV1, PV2;               //principle curvatures
MatrixXi FV;                     //feature line vertices

VectorXd area_map;               //area map, #F x1
VectorXd mean_curv_map;          //mean curvature map, #V x1
VectorXd gaus_curv_map;          //gaussian curvature map, #V x1
MatrixXi control_map;            //control map, pixel width x height
MatrixXi sampling_data;          //sampling result, pixel width x height

void reset_mesh(igl::opengl::glfw::Viewer &viewer);
void map_vertices_to_rectangle(const Eigen::MatrixXd& V, const Eigen::VectorXi& bnd, Eigen::MatrixXd& UV);
void harmonic_parameterization();
void calc_face_index_map();
void calc_area_map();
void calc_mean_curvature_map();
void calc_gaussian_curvature_map();
void calc_control_map(igl::opengl::glfw::Viewer &viewer);
void calc_feature_lines();
void overlay_feature_lines(igl::opengl::glfw::Viewer &viewer, int dim);
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
    options.dihedral_angle_thresholding = 30.0;
    options.overlay_feature_lines = false;
    options.use_inverse_mode = true;
    options.use_area_map = true;
    options.use_mean_curv_map = true;
    options.use_gaus_curv_map = false;
    options.scaling_factor = 1.0;
    options.num_of_samples = V.rows() / 10;

    // Calculate principle curvatures
    cout << "Calculate principle curvatures" << endl;
    MatrixXd PD1, PD2;
    igl::principal_curvature(V, F, PD1, PD2, PV1, PV2);

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

        ImGui::Checkbox("Inverse Mode", &options.use_inverse_mode);

        if (ImGui::CollapsingHeader("Features", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputFloat("dihedral angle", &options.dihedral_angle_thresholding);
            ImGui::Checkbox("Overlay feature lines", &options.overlay_feature_lines);

            if (ImGui::Button("Calculate Feature lines"))
            {
                calc_feature_lines();
                cerr << "Num of feature lines: " << FV.rows() << endl;
            }
        }

        if (ImGui::CollapsingHeader("Parameterization", ImGuiTreeNodeFlags_DefaultOpen))
        {
            if (ImGui::Button("Harmonic"))
            {
                harmonic_parameterization();
                calc_face_index_map();

                viewer.data().clear();
                viewer.data().set_mesh(V_uv, F);
                viewer.data().set_uv(V_uv);
                viewer.core.align_camera_center(V_uv, F);
                viewer.data().show_lines = true;
                
                overlay_feature_lines(viewer, 2);
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
            ImGui::InputInt("samples", &options.num_of_samples);

            if (ImGui::Button("Perform Sampling"))
            {
                cout << control_map.rows() << " x " << control_map.cols() << endl;
                int total_pixels = control_map.rows() * control_map.cols();
                double scale = (double)(total_pixels - options.num_of_samples) * WHITE / control_map.sum();
                cout << "Scale: " << scale << endl;
                for (int i = 0; i < control_map.size(); i++)
                {
                    control_map(i) = (int)((double)control_map(i) * scale);
                }
                sampling();
                cout << "Black pixels after sampling:" << (sampling_data.array() == BLACK).count() << endl;
                render_pixel_img(viewer, sampling_data);
            }
        }

        if (ImGui::Button("Triangulate"))
        {
            MatrixXd UV, H;
            MatrixXi E;
            get_uv_coord_from_pixel_img(sampling_data, UV, E);

            igl::triangle::triangulate(UV, E, H, "a0.005q", V2, F2);

            viewer.data().clear();
            viewer.data().set_mesh(V2, F2);
            viewer.core.align_camera_center(V2, F2);
            viewer.data().show_lines = true;
        }

        if (ImGui::Button("Reproject"))
        {
            //viewer.data().clear();
            //viewer.data().set_mesh(V_uv, F);
            //viewer.data().set_uv(V_uv);
            //viewer.data().set_colors(UV_color);
            //viewer.core.align_camera_center(V_uv, F);
            //viewer.data().show_lines = false;
            //viewer.data().show_texture = false;
            //viewer.data().V_material_specular.setZero();
            //viewer.data().F_material_specular.setZero();
            //viewer.data().V_material_ambient.setZero();
            //viewer.data().F_material_ambient.setZero();

            //// Screen capturing
            //const int width = (int)viewer.core.viewport(2);
            //const int height = (int)viewer.core.viewport(3);

            //MatrixXuc R(width, height);
            //MatrixXuc G(width, height);
            //MatrixXuc B(width, height);
            //MatrixXuc A(width, height);
            //viewer.core.draw_buffer(viewer.data(), false, R, G, B, A);

            //// Crop the boarder
            //int top = 0, bottom = A.rows() - 1, left = 0, right = A.cols() - 1;

            //while (A.row(top).maxCoeff() == A.row(top).minCoeff()) top++;
            //while (A.row(bottom).maxCoeff() == A.row(bottom).minCoeff()) bottom--;
            //while (A.col(left).maxCoeff() == A.col(left).minCoeff()) left++;
            //while (A.col(right).maxCoeff() == A.col(right).minCoeff()) right--;

            //MatrixXi R2(width, height);
            //MatrixXi G2(width, height);
            //MatrixXi B2(width, height);
            //int rows = bottom - top + 1, cols = right - left + 1;
            //R2 = R.block(top, left, rows, cols).cast<int>();
            //G2 = G.block(top, left, rows, cols).cast<int>();
            //B2 = B.block(top, left, rows, cols).cast<int>();

            VectorXd dblA;
            igl::doublearea(V_uv, F, dblA);

            // Calculate barycentric coordinates of the triangulated mesh
            // in the original parameterization
            MatrixXd TA, TB, TC, L;
            VectorXi corresponding_triangle = VectorXi::Zero(V2.rows());
            TA.resize(V2.rows(), 2);
            TB.resize(V2.rows(), 2);
            TC.resize(V2.rows(), 2);
            for (int i = 0; i < V2.rows(); i++)
            {
                //int col = V2(i, 0) * (double)(cols - 1);
                //int row = V2(i, 1) * (double)(rows - 1);

                //Color c(R2(row, col), G2(row, col), B2(row, col));
                //int f = face_idx_map.at(c);
                for (int f = 0; f < F.rows(); f++)
                {
                    Vector2d p0 = V_uv.row(F(f, 0));
                    Vector2d p1 = V_uv.row(F(f, 1));
                    Vector2d p2 = V_uv.row(F(f, 2));
                    Vector2d p = V2.row(i);
                    double s = 1 / dblA[f]*(p0[1]*p2[0] - p0[0]*p2[1] + (p2[1]-p0[1])*p[0] + (p0[0]-p2[0])*p[1]);
                    double t = 1 / dblA[f]*(p0[0]*p1[1] - p0[1]*p1[0] + (p0[1]-p1[1])*p[0] + (p1[0]-p0[0])*p[1]);

                    if (s >= 0 && t >= 0 && (1 - s - t) >= 0)
                    {
                        TA.row(i) << V_uv.row(F(f, 0));
                        TB.row(i) << V_uv.row(F(f, 1));
                        TC.row(i) << V_uv.row(F(f, 2));
                        corresponding_triangle[i] = f;
                        break;
                    }
                }
            }
            igl::barycentric_coordinates(V2, TA, TB, TC, L);

            // Reproject the 2D mesh into 3D
            V3.resize(V2.rows(), 3);
            for (int i = 0; i < V2.rows(); i++)
            {
                int f = corresponding_triangle[i];
                V3.row(i) = V.row(F(f, 0)) * L(i, 0) + V.row(F(f, 1)) * L(i, 1) + V.row(F(f, 2)) * L(i, 2);
            }

            // View the new mesh
            viewer.data().clear();

            viewer.data().set_mesh(V3, F2);
            viewer.core.align_camera_center(V3, F2);
            viewer.data().show_texture = false;
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

    overlay_feature_lines(viewer, 3);
}

void calc_feature_lines()
{
    MatrixXi TT, TTi;
    MatrixXd N;
    igl::triangle_triangle_adjacency(F, TT, TTi);
    igl::per_face_normals(V, F, N);

    // Calculate feature lines
    std::vector<int> feature_line;
    double cos_threshold = std::cos(options.dihedral_angle_thresholding * M_PI / 180.0);
    for (int i = 0; i < F.rows(); i++)
    {
        for (int j = 0; j < 3; j++)
        {
            int f = TT(i, j);

            // Avoid counting the same edge twice
            if (i > f) continue;

            if (f >= 0 && f < F.rows())
            {
                double cos_angle = N.row(i).dot(N.row(f));
                if (cos_angle < cos_threshold)
                {
                    feature_line.push_back(i);
                    feature_line.push_back(j);
                }
            }
        }
    }

    // Generate the feature line vertex list
    int num_edges = feature_line.size() / 2;
    FV.resize(num_edges, 2);
    for (int i = 0; i < num_edges; i++)
    {
        int f = feature_line[2 * i];
        int v = feature_line[2 * i + 1];
        FV.row(i) << F(f, v), F(f, (v + 1) % 3);
    }
}

void overlay_feature_lines(igl::opengl::glfw::Viewer &viewer, int dim)
{
    if (options.overlay_feature_lines)
    {
        MatrixXd P1, P2;

        if (dim == 3)
        {
            igl::slice(V, FV.col(0), igl::colon<int>(0, 2), P1);
            igl::slice(V, FV.col(1), igl::colon<int>(0, 2), P2);
        }
        else
        {
            igl::slice(V_uv, FV.col(0), igl::colon<int>(0, 1), P1);
            igl::slice(V_uv, FV.col(1), igl::colon<int>(0, 1), P2);
        }

        viewer.data().add_edges(P1, P2, Eigen::RowVector3d(1.0, 0.0, 0.0));
    }
}

void harmonic_parameterization()
{
    // Find the open boundary
    VectorXi bnd;
    igl::boundary_loop(F, bnd);

    // Map the boundary to a circle, preserving edge proportions
    map_vertices_to_rectangle(V, bnd, bnd_uv);

    // Harmonic parametrization for the internal vertices
    igl::harmonic(V, F, bnd, bnd_uv, 1, V_uv);
}

void calc_face_index_map()
{
    int r, g, b;
    UV_color.resize(F.rows(), 3);
    r = 0;
    g = 0;
    b = 0;

    for (int f = 0; f < F.rows(); f++)
    {
        UV_color.row(f) << (double)r / 255, (double)g / 255, (double)b / 255;
        Color c(r, g, b);
        face_idx_map[c] = f;

        b += 1;
        if (b == 256)
        {
            b = 0;
            g += 1;
            if (g == 256)
            {
                g = 0;
                r += 1;
            }
        }
        assert(r < 256);
    }
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
        temp = R.cast<double>();
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

    overlay_feature_lines(viewer, 2);
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
    int rows = control_map.rows();
    int cols = control_map.cols();

    sampling_data.resize(rows, cols);
    error_diffusion(control_map, sampling_data);

    // Forcing the boundary points on the final sampling map to be black
    for (int i = 0; i < bnd_uv.rows(); i++)
    {
        int r = rows * bnd_uv(i, 1);
        int c = cols * bnd_uv(i, 0);
        sampling_data(r, c) = BLACK;
    }

    // Forcing the four rectangle vertices to be black
    sampling_data(0, 0) = BLACK;
    sampling_data(0, cols - 1) = BLACK;
    sampling_data(rows - 1, 0) = BLACK;
    sampling_data(rows - 1, cols - 1) = BLACK;
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
        if (options.use_inverse_mode)
        {
            norm = 1 - norm;
        }
        color(i, 0) = norm;
        color(i, 1) = norm;
        color(i, 2) = norm;
    }
}
