#include <opencv2/opencv.hpp>
#include "clany/timer.hpp"
#include "svm.h"

using namespace std;
using namespace clany;
using namespace cv;

int main(/*int argc, char* argv[]*/)
{
    clany::CPUTimer timer;
    float train_data[4][2] = {{501, 10}, {255, 10}, {501, 255}, {10, 501}};
    Mat train_mat(4, 2, CV_32FC1, train_data);

    float labels[4] = {1.0, -1.0, -1.0, -1.0};
    Mat label_mat(4, 1, CV_32FC1, labels);

    cv::SVMParams cv_params;
    cv_params.svm_type = cv::SVM::C_SVC;
    cv_params.kernel_type = cv::SVM::LINEAR;
    cv_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    cv::SVM cv_svm;
    cv_svm.train(train_mat, label_mat, Mat(), Mat(), cv_params);

    ml::SVM svm;
    svm.train(train_mat, label_mat, cv_params.svm_type, cv_params.kernel_type);
    timer.delta("Train time");

    //////////////////////////////////////////////////////////////////////////
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);
    Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255);
    // Show the decision regions given by the SVM
    for (int i = 0; i < image.rows; ++i)
    for (int j = 0; j < image.cols; ++j) {
        //ml::SVMNode sample[] = {{0, j}, {1, i}, {-1, 0}};
        Mat sample = (Mat_<float>(1, 2) << j, i);
        double response = svm.predict(sample);

        if (response == 1)
            image.at<Vec3b>(i, j) = green;
        else if (response == -1)
            image.at<Vec3b>(i, j) = blue;
        else
            image.at<Vec3b>(i, j) = red;
    }
    timer.delta("Predict time");

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
    circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);
}