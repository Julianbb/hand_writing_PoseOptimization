#include <Eigen/Eigen>
//#include <core/eigen.hpp>
#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tic_toc.h"
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/base_unary_edge.h>

using namespace std;
using namespace Eigen;

typedef Matrix<double, 6, 6> Matrix_66;
typedef Matrix<double, 2, 6> Matrix_26;
typedef Matrix<double, 6, 1> Vector6d;



cv::Mat Eigen2Mat_44(const Eigen::Matrix<double,4,4> &src)
{
    cv::Mat dst(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            dst.at<float>(i,j)=src(i,j);

    return dst.clone();
}



Matrix<double, 4, 4>  Mat2Eigen_44(const cv::Mat& src) 
{
    Matrix<double, 4, 4> dst;
    
    for(int i = 0; i <4; i++)
    {
        for(int j = 0; j <4; j++)
        {
            dst(i,j) = src.at<float>(i,j);
        }
    }
    return dst;
}


Matrix<double, 3, 3>  Mat2Eigen_33(const cv::Mat& src) 
{
    Matrix<double, 3, 3> dst;
    for(int i = 0; i <3; i++)
    {
        for(int j = 0; j <3; j++)
        {
            dst(i,j) = src.at<float>(i,j);
        }
    }

    return dst;
}




Matrix3d SO3hat(const Vector3d & v)
{
    Matrix3d Omega;
    Omega <<  0, -v(2),  v(1)
        ,  v(2),     0, -v(0)
        , -v(1),  v(0),    0;
    return Omega;
}


//旋转在前，平移在后
void exp(const Vector6d & update, Matrix3d& R_, Vector3d& t_)
{
    Vector3d omega;
    for (int i=0; i<3; i++)
    omega[i]=update[i];
    Vector3d upsilon;
    for (int i=0; i<3; i++)
    upsilon[i]=update[i+3];

    double theta = omega.norm();
    Matrix3d Omega = SO3hat(omega);

    Matrix3d R;
    Matrix3d V;
    if (theta<0.00001)
    {
    //TODO: CHECK WHETHER THIS IS CORRECT!!!
    R = (Matrix3d::Identity() + Omega + Omega*Omega);

    V = R;
    }
    else
    {
    Matrix3d Omega2 = Omega*Omega;

    R = (Matrix3d::Identity()
        + sin(theta)/theta *Omega
        + (1-cos(theta))/(theta*theta)*Omega2);

    V = (Matrix3d::Identity()
        + (1-cos(theta))/(theta*theta)*Omega
        + (theta-sin(theta))/(pow(theta,3))*Omega2);
    }

    R_ = R;
    t_ = V*upsilon;
   
}



Vector2d ComputeError(const Vector2d& obs, const Matrix3d& Rcw,  const Vector3d& tcw, const Vector3d& Pw, const Matrix3d& K)
{
    Vector2d error;
    Vector3d local_point = Rcw*Pw + tcw;
    local_point = local_point/local_point[2];

    error = obs - (K * local_point).head<2>();
    return error; 
}

double ComputeChi2(const Vector2d& error, const Matrix2d& info)
{
    return error.dot(info*error);
}

Matrix_26 ComputeJacobian(const Matrix3d& Rcw,  const Vector3d& tcw, const Vector3d& Pw, const Matrix3d& K)
{

  Matrix_26 jacobian;
  Vector3d local_point = Rcw * Pw + tcw;

    double fx = K(0,0);
    double fy = K(1,1);
    double cx = K(0,2);
    double cy = K(1,2);

    double x = local_point[0];
    double y = local_point[1];
    double invz = 1.0/local_point[2];
    double invz_2 = invz*invz;

    jacobian(0,0) =  x*y*invz_2 *fx;
    jacobian(0,1) = -(1+(x*x*invz_2)) *fx;
    jacobian(0,2) = y*invz *fx;
    jacobian(0,3) = -invz *fx;
    jacobian(0,4) = 0;
    jacobian(0,5) = x*invz_2 *fx;

    jacobian(1,0) = (1+y*y*invz_2) *fy;
    jacobian(1,1) = -x*y*invz_2 *fy;
    jacobian(1,2) = -x*invz *fy;
    jacobian(1,3) = 0;
    jacobian(1,4) = -invz *fy;
    jacobian(1,5) = y*invz_2 *fy;

    return jacobian;
}





void  MakeHessian(const vector<Vector2d>& obs,
                  const vector<Vector2d>& error, 
                  const vector<Vector3d>& Pws, 
                  const Matrix3d& Rcw,
                  const Vector3d& tcw, 
                  const Matrix3d& K,
                  const vector<Matrix2d>& info_matrix,
                  Matrix<double, 6, 6>& H,
                   Matrix<double, 6, 1>& b)
{
    b = Vector6d::Zero(); // 6*1
    H = Matrix_66::Zero(); // 6*6

    for(int i=0; i< Pws.size(); i++)
    {
        Matrix_26 jabobian = ComputeJacobian(Rcw, tcw, Pws[i], K);

        MatrixXd Jt = jabobian.transpose(); // * info_matrix[i];

        b -= Jt*error[i];

        H += Jt * jabobian;

    }
}





bool IsGoodStepInLM(double& lambda,  
                    double& ni, 
                    const Vector6d& delta_x,
                    double& chi2, 
                    const Vector6d& b,                    
                    vector<Vector2d>& error,
                    const vector<Vector2d>& obs,
                    const vector<Vector3d>& Pws, 
                    const Matrix3d& Rcw, 
                    const Vector3d& tcw, 
                    const Matrix3d& K,
                    const vector<Matrix2d>& info_matrix 
                    )
                    
{
    double scale = 0.;
    scale = 0.5 * delta_x.transpose() * (lambda * delta_x + b);
    scale += 1e-3; // 防止为0


    double tmp_chi2 =0.;
    for(int i =0; i< error.size(); i++)
    {
        Vector2d err = ComputeError(obs[i], Rcw, tcw, Pws[i], K);
        error[i] = err;
    
        tmp_chi2 += ComputeChi2(error[i], info_matrix[i]);
    }

    double rho = (chi2 - tmp_chi2) / scale;
    if(rho > 0) //证明上次的迭代误差是下降的
    {
        double alpha = 1. - std::pow(2*rho-1, 3);
        alpha = std::min(alpha, 2. / 3.);
        double tmp_factor = std::max(1./3, alpha);
        
        lambda *= tmp_factor;
        ni = 2.0;

        chi2 = tmp_chi2; //跟新误差

        return true;
    }
    else
    {
        lambda = lambda * ni;
        ni = 2*ni;
        return false;
    }

}

void UpdateStates(Matrix<double, 3, 3>& Rcw, Vector3d& tcw, const Matrix<double, 6, 1>& delta_x)
{
    Matrix3d R_tmp;
    Vector3d t_tmp;
    exp(delta_x, R_tmp, t_tmp);

    Rcw = R_tmp * Rcw;
    tcw = R_tmp * tcw + t_tmp;
}


void RollbackStates(Matrix<double, 3, 3>& Rcw, Vector3d& tcw, const Matrix<double, 6, 1>& delta_x)
{
    Matrix3d R_tmp;
    Vector3d t_tmp;
    exp(-delta_x, R_tmp, t_tmp);

    Rcw = R_tmp * Rcw;
    tcw = R_tmp * tcw + t_tmp;
}

void SolveLinearSystem(const Matrix<double, 6, 6>& H, const Matrix<double, 6, 1>& b,  Matrix<double, 6, 1>& delta_x,  double lambda)
{
        MatrixXd tmp_H = H;

        for(int i = 0; i <H.rows(); i++) // 解方程对角线+lamda
            tmp_H(i,i) += lambda;
        
        delta_x = tmp_H.llt().solve(b);
        

}

void LM_Solver(int iterations, 
                const vector<Vector3d>& Pws, 
                const vector<Vector2d>& obs, 
                const vector<Matrix2d>& info_matrix,
                vector<Vector2d>& error,
                Matrix3d& Rcw,
                Vector3d& tcw,
                const Matrix3d& K)
{
    double lambda = 1.0;
    double ni = 2.0;
    double stop_chi2 = 0.0;
    double chi2 = 0.0;

    Vector6d b = Vector6d::Zero(); // 6*1
    Matrix_66 H = Matrix_66::Zero(); // 6*6
    Vector6d delta_x = Vector6d::Zero();

 
    error.resize(Pws.size());
    // 计算初始error
    for(int i =0; i< Pws.size(); i++)
    {
        Vector2d err = ComputeError(obs[i], Rcw, tcw, Pws[i], K);
        error[i] = err;
    }

    //计算初始　Chi2
    for(int i = 0; i < error.size(); i++)
    {
        chi2 += ComputeChi2(error[i], info_matrix[i]);
    }
    stop_chi2 = 1e-6 * chi2;  //TODO:自行修改，误差下降了1e-6倍，则停止


    MakeHessian(obs, error, Pws, Rcw, tcw, K, info_matrix, H, b);

    double maxDiagnal = 0.;
    ulong size = H.rows();
    for(int i = 0; i <size; i++)
        maxDiagnal = std::max(std::fabs(H(i,i)), maxDiagnal);

    lambda = 1e-5 * maxDiagnal;  // lamda初始值　= 1e-5 * H对角元素的最大值
    
//============================ LM 初始化完成 =======================================

    bool stop = false;
    int its = 0;

    while(!stop && (its < iterations))
    {
        cout << " iters: " << its << " Chi2 : " << chi2  << " lambda: " << lambda << endl;
        bool oneStepSucceed = false;
        int false_cnt = 0;
        while(!oneStepSucceed)
        {
            SolveLinearSystem(H, b, delta_x, lambda); 
            if( false_cnt > 10)  // TODO:根据需要修改控制delta_x的模小于一定量级就停止 delta_x.squaredNorm() < 1e-12 ||
            {
                //cout << false_cnt << endl;
                stop = true;
                break;
            }
            UpdateStates(Rcw, tcw, delta_x);
            oneStepSucceed = IsGoodStepInLM(lambda, ni, delta_x, chi2, b, error, obs, Pws, Rcw, tcw, K, info_matrix);

            if(oneStepSucceed)
            {
                MakeHessian(obs, error, Pws, Rcw, tcw, K, info_matrix, H, b);
                
                false_cnt = 0;
            }
            else
            {
                false_cnt++;
                RollbackStates(Rcw, tcw, delta_x); // 误差没下降，回滚
            }
        }

        its++;
        if(chi2 < stop_chi2) 
        {
            stop = true;
        }
            
    }
}



// void PoseOptimization(Frame *pFrame)
// {
//     Matrix<double, 3, 4> Tcw;
//     Matrix<double, 3, 3> K;
  
//     cv2eigen(pFrame->mTcw, Tcw); 
//     cv2eigen(pFrame->mK, K); 
//     Matrix3d R = Tcw.leftCols(3);
//     Vector3d t = Tcw.rightCols(1);

//     const int N = pFrame->N;

//     vector<Vector3d> Pws; Pws.reserve(N);
//     vector<Vector2d> obs; obs.reserve(N);
//     vector<Matrix2d> info_matrix; info_matrix.reserve(N);

//     for(int i=0; i<N; i++)
//     {
//         MapPoint* pMP = pFrame->mvpMapPoints[i];
//         if(pMP)
//         {
//             Eigen::Vector2d obs_tmp;
//             const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
//             obs_tmp << kpUn.pt.x, kpUn.pt.y;
//             obs.push_back(obs_tmp);

//             const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
//             Matrix2d info_matrix_tmp = Matrix2d::Identity()*invSigma2;
//             info_matrix.push_back(info_matrix_tmp);

//             cv::Mat Xw = pMP->GetWorldPos();
//             Vector3d Pws_tmp;
//             Pws_tmp[0] = Xw.at<float>(0);
//             Pws_tmp[1] = Xw.at<float>(1);
//             Pws_tmp[2] = Xw.at<float>(2);
//             Pws.push_back(Pws_tmp);
//         }

//     }

//     LM_Solver(10, Pws, obs, info_matrix, R, t, K);

//     Matrix<double, 3, 4> Tcw_after_optimization;
//     Tcw_after_optimization.leftCols(3) = R;
//     Tcw_after_optimization.rightCols(1) = t;
//     cv::Mat pose;
//     eigen2cv(Tcw_after_optimization ,pose);
//     pFrame->SetPose(pose);

// }



void g2oOptimize(int iterations, 
                const vector<Vector3d>& Pws, 
                const vector<Vector2d>& obs, 
                const vector<Matrix2d>& info_matrix,
                Matrix3d& Rcw,
                Vector3d& tcw,
                const Matrix3d& K)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType>(linearSolver));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<g2o::BlockSolver_6_3>(solver_ptr));     
    optimizer.setAlgorithm(solver);


    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(g2o::SE3Quat(Rcw, tcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);


    const int N = Pws.size();

    for(int i = 0; i <N; i++)
    {
        g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e->setMeasurement(obs[i]);
        e->setInformation(info_matrix[i]);
        e->fx = K(0,0);
        e->fy = K(1,1);
        e->cx = K(0,2);
        e->cy = K(1,2);

        e->Xw = Pws[i];
        optimizer.addEdge(e);
    }

    optimizer.setVerbose(true);
    optimizer.initializeOptimization(0);
    optimizer.optimize(iterations);

    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    Rcw = SE3quat_recov.rotation().toRotationMatrix();
    tcw = SE3quat_recov.translation();
}




// 生成1个pose, N个点
void GenerateData(int N, const Matrix3d& K, vector<Vector3d>& Pws, vector<Vector2d>& obs, vector<Matrix2d>& info_matrix)
{
    
    Matrix3d Rcw = AngleAxisd(M_PI/4, Vector3d::UnitZ()).toRotationMatrix();;
    Vector3d tcw = Vector3d(0,0,1); 

    default_random_engine e;
    std::uniform_real_distribution<double> xy_rand(-4, 4);
    std::uniform_real_distribution<double> z_rand(4, 8);
    for(int i = 0; i < N; i++)
    {
        Pws.push_back(Vector3d(xy_rand(e), xy_rand(e), z_rand(e)));
    }


    // generate features frome Pws
    for(int i = 0; i <N; i++)
    {
        Vector3d local = Rcw * Pws[i] + tcw;
        local  = local/local[2];
        local = K * local;
        obs.push_back(Vector2d(local[0], local[1]));
    }

    //信息矩阵
    for(int i=0; i<N; i++)
    {
        info_matrix[i] = Matrix2d::Identity();
    }
}








int main(void)
{
    int N =200;
    vector<Vector3d> Pws; Pws.reserve(N); // 地图点
    vector<Vector2d> obs; obs.reserve(N); // 观测
    vector<Matrix2d> info_matrix; info_matrix.reserve(N);
    vector<Vector2d> error; error.reserve(N);
    Matrix3d K = Matrix3d::Identity();
    K(0,0) = 640/20;
    K(1,1) = 320/20;
    K(0,2) = 320;
    K(1,2) = 160;

    //生成真实数据
    GenerateData(N, K, Pws, obs, info_matrix);





    // 在R t加噪声，　R用 M_PI/4生成的数据，优化前用M_PI/2作为噪声数据
    Matrix3d R = AngleAxisd(M_PI/2, Vector3d::UnitZ()).toRotationMatrix();;
    Vector3d t = Vector3d(0,0,1); 

    default_random_engine generator;
    normal_distribution<double> noise_t_pdf(1., 1.);
    t += Vector3d(noise_t_pdf(generator), noise_t_pdf(generator), noise_t_pdf(generator));





    cout << "before optimization " <<endl;
    cout << R << endl;
    cout << t << endl;
    cout << endl << endl;


    TicToc solver_tic;
    LM_Solver(10, Pws, obs, info_matrix, error, R, t, K); // LM_Solver  g2oOptimize
    cout << solver_tic.toc() << " ms " << endl;


    cout << endl << endl;
    cout << "after optimization " <<endl;
    cout << R << endl;
    cout << t << endl;

}

