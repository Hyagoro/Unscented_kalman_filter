#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.0;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1.0;

    //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

    //set state dimension
    n_x_ = x_.size();

    //set measurement dimension, radar can measure r, phi, and r_dot
    n_z_radar_ = 3;
    n_z_lidar_ = 2;

    R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
    R_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);

    n_aug_ = 7;

    lambda_ = 3 - n_aug_;

    weights_ = VectorXd(2 * n_aug_ + 1);

    // predicted sigma points matrix
    Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
}

UKF::~UKF() = default;

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {

    if (!is_initialized_) {

        //covariance matrix
        P_ << 1, 0, 0, 0, 0,
                0, 1, 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 0, 1;

        //measurement noise covariance matrix
        R_radar_ << std_radr_ * std_radr_, 0, 0,
                0, std_radphi_ * std_radphi_, 0,
                0, 0, std_radrd_ * std_radrd_;

        R_lidar_ << std_laspx_ * std_laspx_, 0,
                0, std_laspy_ * std_laspy_;


        // set weights
        double weight_0 = lambda_ / (lambda_ + n_aug_);
        weights_(0) = weight_0;
        for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
            double weight = 0.5 / (n_aug_ + lambda_);
            weights_(i) = weight;
        }

        is_initialized_ = true;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            float rho = measurement_pack.raw_measurements_[0];
            float cos_phi = cos(measurement_pack.raw_measurements_[1]);
            float sin_phi = sin(measurement_pack.raw_measurements_[1]);
            float rho_dot = measurement_pack.raw_measurements_[2];
            //x, y, vx, vy
            x_ << rho * cos_phi, rho * sin_phi, rho_dot * cos_phi, rho_dot * sin_phi, 0;
        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
            */
            x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;

        }
        time_us_ = measurement_pack.timestamp_;

        return;
    }

    //compute the time elapsed between the current and previous measurements
    float dt = (measurement_pack.timestamp_ - time_us_) / 1000000.0;    //dt - expressed in seconds
    time_us_ = measurement_pack.timestamp_;

    Prediction(dt);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
        UpdateRadar(measurement_pack);

    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
        UpdateLidar(measurement_pack);
    }
}

void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out) {
    //create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    Xsig_aug.fill(0.0);

    //create augmented mean state
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
        Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
    }

    *Xsig_out = Xsig_aug;
}


void UKF::GenerateSigmaPoints(MatrixXd *Xsig_out) {
    //create sigma point matrix
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

    //calculate square root of P (covariance matrix)
    MatrixXd A = P_.llt().matrixL();

    //set first column of sigma point matrix
    Xsig.col(0) = x_;

    //set remaining sigma points
    for (int i = 0; i < n_x_; i++) {
        Xsig.col(i + 1) = x_ + sqrt(lambda_ + n_x_) * A.col(i);
        Xsig.col(i + 1 + n_x_) = x_ - sqrt(lambda_ + n_x_) * A.col(i);
    }

    *Xsig_out = Xsig;
}

void UKF::SigmaPointPrediction(MatrixXd Xsig_aug, double delta_t) {
//    std::cout << "delta_t" << std::endl;
//    std::cout << delta_t << std::endl;

    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }
}

void UKF::PredictMeanAndCovariance(VectorXd *x_out, MatrixXd *P_out) {
    //create vector for predicted state
    VectorXd x = VectorXd(n_x_);

    //create covariance matrix for prediction
    MatrixXd P = MatrixXd(n_x_, n_x_);

    //predicted state mean
    x.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x = x + weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

//        std::cout << "Xsig_pred_" << std::endl;
//        std::cout << Xsig_pred_ << std::endl;
//
//        std::cout << "x" << std::endl;
//        std::cout << x << std::endl;
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x;

//        std::cout << "x_diff" << std::endl;
//        std::cout << x_diff << std::endl;
        //angle normalization
        while (x_diff(3) > M_PI) {
            float toto = x_diff(3);
            toto -= 2. * M_PI;
            x_diff(3) = toto;
        }
        while (x_diff(3) < -M_PI) {
            float toto = x_diff(3);
            toto += 2. * M_PI;
            x_diff(3) = toto;
        }

        P = P + weights_(i) * x_diff * x_diff.transpose();
    }

    *x_out = x;
    *P_out = P;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

    MatrixXd Xsig = MatrixXd(5, 11);
    GenerateSigmaPoints(&Xsig);
    AugmentedSigmaPoints(&Xsig);


    SigmaPointPrediction(Xsig, delta_t);


    PredictMeanAndCovariance(&x_, &P_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

    //TODO avoid code duplication
    //------------
    // Step 1
    //------------

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z_lidar_, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        Zsig(0, i) = Xsig_pred_(0, i);
        Zsig(1, i) = Xsig_pred_(1, i);
    }


    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z_lidar_);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z_lidar_, n_z_lidar_);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    S = S + R_lidar_;

    //------------
    // Step 2
    //------------

    VectorXd z = meas_package.raw_measurements_;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z_lidar_);


    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < n_aug_; i++) {  //2n+1 sigma points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    //TODO
    MatrixXd NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
    std::cout << "NIS_laser_" << std::endl;
    std::cout << NIS_laser_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

    //------------
    // Step 1
    //------------

    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z_radar_, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        // extract values for better readability
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                           //roh
        Zsig(1, i) = atan2(p_y, p_x);                                       //phi
        Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);   //roh_dot
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z_radar_);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    S = S + R_radar_;

    //------------
    // Step 2
    //------------

    VectorXd z = meas_package.raw_measurements_;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);


    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 sigma points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        //angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // TODO write in file and plot it in python
    MatrixXd NIS_radar = z_diff.transpose() * S.inverse() * z_diff;
    std::cout << "NIS_radar" << std::endl;
    std::cout << NIS_radar << std::endl;
}
