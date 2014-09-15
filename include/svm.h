#ifndef SVM_H
#define SVM_H

#include <memory>
#include <opencv2/opencv.hpp>
#include "clany/clany_macros.h"

#define _DENSE_REP
#include "libsvm.h"

_CLANY_BEGIN
namespace ml {

using SVMNode = svm_node;
using SVMProblem = svm_problem;
using SVMModel = svm_model;
using SVMParameter = svm_parameter;

struct SVMModelDeletor
{
    void operator()(SVMModel* ptr)
    {
        svm_free_and_destroy_model(&ptr);
    }
};


class SVM
{
public:
    enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
    enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

    void train(const cv::Mat& train_data, cv::InputArray labels, const SVMParameter& params);
    void train(const cv::Mat& train_data, cv::InputArray labels,
               int svm_type = C_SVC, int kernel_type = RBF, int k_fold = 10);

    double predict(cv::InputArray sample) const;
    double predict(const SVMNode& sample) const;

    double predict(cv::InputArray sample, vector<double>& vals, bool return_prob_val = false) const;
    double predict(const SVMNode& sample, vector<double>& vals, bool return_prob_val = false) const;

private:
    cv::Mat train_mat;
    cv::Mat label_mat;
    SVMParameter params;
    unique_ptr<SVMModel, SVMModelDeletor> model_ptr;
};

} // End namespace ml
_CLANY_END

#endif // SVM_H