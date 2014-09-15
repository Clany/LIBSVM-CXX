#include <iostream>
#include "svm.h"

using namespace std;
using namespace clany;
using namespace ml;
using cv::Mat;
using cv::InputArray;


void SVM::train(const Mat& train_data, InputArray labels, const SVMParameter& _params)
{
    train_data.convertTo(train_mat, CV_64F);
    labels.getMat().convertTo(label_mat, CV_64F);
    params = _params;

    vector<SVMNode> data_node(train_mat.rows);
    int ft_dim = train_mat.cols;
    for (int i = 0; i < data_node.size(); ++i) {
        data_node[i].dim = ft_dim;
        data_node[i].values = train_mat.ptr<double>(i);
    }

    SVMProblem samples;
    samples.l = train_mat.rows;
    samples.y = label_mat.ptr<double>();
    samples.x = data_node.data();

    model_ptr.reset(svm_train(&samples, &params));
}


void SVM::train(const Mat& train_data, InputArray labels, int svm_type, int kernel_type, int k_fold)
{
    Mat cv_train_mat, cv_label_mat;
    train_data.convertTo(cv_train_mat, CV_32F);
    labels.getMat().convertTo(cv_label_mat, CV_32F);

    CvSVMParams svm_params;
    svm_params.svm_type = CvSVM::C_SVC;
    svm_params.kernel_type = CvSVM::RBF;

    CvSVM svm;
    svm.train_auto(cv_train_mat, cv_label_mat, Mat(), Mat(), svm_params, k_fold);

    svm_params = svm.get_params();
    params.C = svm_params.C;
    params.coef0 = svm_params.coef0;
    params.degree = svm_params.degree;
    params.gamma = svm_params.gamma;
    params.nu = svm_params.nu;
    params.p = svm_params.p;
    params.eps = svm_params.term_crit.epsilon;

    train(train_data, labels, params);
}


double SVM::predict(cv::InputArray _sample) const
{
    Mat sample = _sample.getMat();
    if (sample.type() != CV_64FC1) sample.convertTo(sample, CV_64F);

    SVMNode data_node;
    data_node.dim = sample.rows*sample.cols;
    data_node.values = sample.ptr<double>();

    return svm_predict(model_ptr.get(), &data_node);
}


double SVM::predict(const SVMNode& sample) const
{
    return svm_predict(model_ptr.get(), &sample);
}


double SVM::predict(cv::InputArray _sample, vector<double>& vals, bool return_prob_val) const
{
    Mat sample = _sample.getMat();
    if (sample.type() != CV_64FC1) sample.convertTo(sample, CV_64F);

    SVMNode data_node;
    data_node.dim = sample.rows*sample.cols;
    data_node.values = sample.ptr<double>();

    return predict(data_node, vals, return_prob_val);
}


double SVM::predict(const SVMNode& sample, vector<double>& vals, bool return_prob_val) const
{
    vals.resize(model_ptr->nr_class);
    if (return_prob_val) {
        return svm_predict_probability(model_ptr.get(), &sample, vals.data());
    }
    return svm_predict_values(model_ptr.get(), &sample, vals.data());
}