#include <hw_tool.h>

using namespace std;
using namespace Eigen;

void Homeworktool::initGridMap(double _resolution, Vector3d global_xyz_l, Vector3d global_xyz_u, int max_x_id, int max_y_id, int max_z_id)
{   
    gl_xl = global_xyz_l(0);
    gl_yl = global_xyz_l(1);
    gl_zl = global_xyz_l(2);

    gl_xu = global_xyz_u(0);
    gl_yu = global_xyz_u(1);
    gl_zu = global_xyz_u(2);
    
    GLX_SIZE = max_x_id;
    GLY_SIZE = max_y_id;
    GLZ_SIZE = max_z_id;
    GLYZ_SIZE  = GLY_SIZE * GLZ_SIZE;
    GLXYZ_SIZE = GLX_SIZE * GLYZ_SIZE;

    resolution = _resolution;
    inv_resolution = 1.0 / _resolution;    

    data = new uint8_t[GLXYZ_SIZE];
    memset(data, 0, GLXYZ_SIZE * sizeof(uint8_t));
}

void Homeworktool::setObs(const double coord_x, const double coord_y, const double coord_z)
{   
    if( coord_x < gl_xl  || coord_y < gl_yl  || coord_z <  gl_zl || 
        coord_x >= gl_xu || coord_y >= gl_yu || coord_z >= gl_zu )
        return;

    int idx_x = static_cast<int>( (coord_x - gl_xl) * inv_resolution);
    int idx_y = static_cast<int>( (coord_y - gl_yl) * inv_resolution);
    int idx_z = static_cast<int>( (coord_z - gl_zl) * inv_resolution);      
    
    data[idx_x * GLYZ_SIZE + idx_y * GLZ_SIZE + idx_z] = 1;
}

bool Homeworktool::isObsFree(const double coord_x, const double coord_y, const double coord_z)
{
    Vector3d pt;
    Vector3i idx;
    
    pt(0) = coord_x;
    pt(1) = coord_y;
    pt(2) = coord_z;
    idx = coord2gridIndex(pt);

    int idx_x = idx(0);
    int idx_y = idx(1);
    int idx_z = idx(2);

    return (idx_x >= 0 && idx_x < GLX_SIZE && idx_y >= 0 && idx_y < GLY_SIZE && idx_z >= 0 && idx_z < GLZ_SIZE && 
           (data[idx_x * GLYZ_SIZE + idx_y * GLZ_SIZE + idx_z] < 1));
}

Vector3d Homeworktool::gridIndex2coord(const Vector3i & index) 
{
    Vector3d pt;

    pt(0) = ((double)index(0) + 0.5) * resolution + gl_xl;
    pt(1) = ((double)index(1) + 0.5) * resolution + gl_yl;
    pt(2) = ((double)index(2) + 0.5) * resolution + gl_zl;

    return pt;
}

Vector3i Homeworktool::coord2gridIndex(const Vector3d & pt) 
{
    Vector3i idx;
    idx <<  min( max( int( (pt(0) - gl_xl) * inv_resolution), 0), GLX_SIZE - 1),
            min( max( int( (pt(1) - gl_yl) * inv_resolution), 0), GLY_SIZE - 1),
            min( max( int( (pt(2) - gl_zl) * inv_resolution), 0), GLZ_SIZE - 1);                  
  
    return idx;
}

Eigen::Vector3d Homeworktool::coordRounding(const Eigen::Vector3d & coord)
{
    return gridIndex2coord(coord2gridIndex(coord));
}

double Homeworktool::OptimalBVP(Eigen::Vector3d _start_position,Eigen::Vector3d _start_velocity,Eigen::Vector3d _target_position)
{
    double delt_px,delt_py,delt_pz;
    double delt_vx,delt_vy,delt_vz;
    delt_px = _target_position(0) - _start_position(0);
    delt_py = _target_position(1) - _start_position(1);
    delt_pz = _target_position(2) - _start_position(2);
    delt_vx = -_start_velocity(0);
    delt_vy = -_start_velocity(1);
    delt_vz = -_start_velocity(2);
    double vx_0 = _start_velocity(0);
    double vy_0 = _start_velocity(1);
    double vz_0 = _start_velocity(2);

    // dJ=T^4+24*(delt_px*delt_vx+delt_py*delt_vy+delt_pz*delt_vz)*T-36*(pow(delt_px,2)+pow(delt_py,2)+pow(delt_pz,2)-4*(pow(delt_vx,2)+pow(delt_vy,2)+pow(delt_vz,2)*T^2));
    double alpha=-12 * (delt_vx *vx_0+delt_vy*vy_0+delt_vz*vz_0
                                +pow(vx_0,2)+pow(vy_0,2)+pow(vz_0,2))
                                -4*(pow(delt_vx,2)+pow(delt_vy,2)+pow(delt_vz,2));
    double beta=24*(delt_px*delt_vx+delt_py*delt_vy+delt_pz*delt_vz)
                             +48*(delt_px*vx_0+delt_py*vy_0+delt_pz*vz_0);
    double theta =-36*(pow(delt_px,2)+pow(delt_py,2)+pow(delt_pz,2));
    // dJ =t^4+0*t ^3+alpha*t^2+beta*t
    /*伴随矩阵求解一元四次方程
    P(X)=X^n+a0*X^n-1+a1*X^n-2....+an-2*X+an-1
    伴随矩阵为
    Mx=[0 0 0 ......0  -an-1]
              1 0 0 ......0  -an-2
              0 1 0 ......0  -an-3
              ..............
              0 0 0......1    -a0
    */
    Eigen::Matrix<double, 4, 4> matrix_44;
    Eigen::Matrix<complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix_eigenvalues;
    matrix_44 << 0, 0, 0, -theta,
                                1, 0, 0, -beta,
                                0, 1, 0, -alpha,
                                0, 0, 1, 0;
    matrix_eigenvalues = matrix_44.eigenvalues();
  Eigen::MatrixXd eigenvalues_real =matrix_eigenvalues.real();
  Eigen::MatrixXd eigenvalues_image=matrix_eigenvalues.imag();
  std::vector<double> q;
  for (int i = 0; i < matrix_eigenvalues.size(); i++)
  {
    if (eigenvalues_image(i)==0&&eigenvalues_real(i)>0)
    {
      q.push_back(eigenvalues_real(i));
    }
  }
  double t=q[0];
  double coeff1=12*(pow(vx_0,2)+pow(vy_0,2)+pow(vz_0,2)
                               +vx_0*delt_vx+vy_0*delt_vy+vz_0*delt_vz)
                               +4*(pow(delt_vx,2)+pow(delt_vy,2)+pow(delt_vz,2));
  double coeff2=-24*(delt_px*vx_0+delt_py*vy_0+delt_pz*vz_0)
                                  -12*(delt_px*delt_vx+delt_py*delt_vy+delt_pz*delt_vz);
  double coeff3=12*(pow(delt_px,2)+pow(delt_py,2)+pow(delt_pz,2));
  double optimal_cost = t+coeff1/t+coeff2/pow(t,2)+coeff3/pow(t,3);
    /*
    STEP 2: go to the hw_tool.cpp and finish the function Homeworktool::OptimalBVP
    the solving process has been given in the document
    because the final point of trajectory is the start point of OBVP, so we input the pos,vel to the OBVP
    after finish Homeworktool::OptimalBVP, the Trajctory_Cost will record the optimal cost of this trajectory
    */
    return optimal_cost;
}
