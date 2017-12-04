#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    if(estimations.empty() || estimations.size() != ground_truth.size()){
        return rmse;
    }

    //accumulate squared residuals
    unsigned long es_size = estimations.size();
    for(unsigned long i=0; i < es_size; ++i){
        VectorXd toto = estimations.at(i).array() - ground_truth.at(i).array();
        rmse = rmse.array() + (toto.array() * toto.array());
    }
    rmse = rmse.array() / es_size;
    return rmse.array().sqrt();
}