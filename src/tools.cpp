#include <iostream>
#include "tools.h"

#define EPS 0.0001 
#define EPS2 0.0000001

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  if(estimations.size() == 0){
    cout << "Input is empty" << endl;
    return rmse;
  }
  
  if(estimations.size() != ground_truth.size()){
    cout << "Invalid estimation or ground_truth. Data should have the same size" << endl;
    return rmse;
  }
 
  for(unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  MatrixXd Hj(3,4);
  
  if (fabs(px) < EPS and fabs(py) < EPS){
	  px = EPS;
	  py = EPS;
  }
  
  float c1 = px*px+py*py;
  
  if(fabs(c1) < EPS2){
	c1 = EPS2;
  }
  float c2 = sqrt(c1);
  float c3 = (c1*c2);
  
  Hj << (px/c2), (py/c2), 0, 0,
       -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
  return Hj;
}
