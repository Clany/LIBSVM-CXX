/////////////////////////////////////////////////////////////////////////////////
// The MIT License(MIT)
//
// Copyright (c) 2014 by Tiangang Song
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
/////////////////////////////////////////////////////////////////////////////////

#ifndef SVM_H
#define SVM_H

#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include "clany/clany_defs.h"
#include "libsvm.h"

_CLANY_BEGIN
namespace ml {
using SVMNode      = svm_node;
using SVMProblem   = svm_problem;
using SVMModel     = svm_model;

struct SVMParameter : svm_parameter {
#ifdef CV_VERSION
    SVMParameter& operator= (const cv::SVMParams& params) {
        svm_type    = params.svm_type;
        kernel_type = params.kernel_type;
        C           = params.C;
        coef0       = params.coef0;
        degree      = static_cast<int>(params.degree);
        gamma       = params.gamma;
        nu          = params.nu;
        p           = params.p;
        eps         = params.term_crit.epsilon;
        max_iter    = params.term_crit.max_iter;

        return *this;
    }

    operator cv::SVMParams() const {
        cv::SVMParams params(svm_type, kernel_type, degree, gamma,
                             coef0, C, nu, p, nullptr,
                             cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, max_iter, eps));
        return params;
    }
#endif
};

struct SVMParamGrid {
    enum { C = 0, GAMMA = 1, P = 2, NU = 3, COEF = 4, DEGREE = 5 };

    SVMParamGrid() = default;
    SVMParamGrid(double min_val, double max_val, double log_step)
        : min_val(min_val), max_val(max_val), step(log_step) {}

    bool isValid() {
        return min_val > 0 && max_val > min_val && step >= 1;
    }

    static auto getDefault(int grid_id) -> SVMParamGrid {
        SVMParamGrid grid;

        switch (grid_id) {
        case C:
            grid.min_val = 0.1;
            grid.max_val = 500;
            grid.step = 5; // total iterations = 5
            break;
        case GAMMA:
            grid.min_val = 1e-5;
            grid.max_val = 0.6;
            grid.step = 15; // total iterations = 4
            break;
        case P:
            grid.min_val = 0.01;
            grid.max_val = 100;
            grid.step = 7; // total iterations = 4
            break;
        case NU:
            grid.min_val = 0.01;
            grid.max_val = 0.2;
            grid.step = 3; // total iterations = 3
            break;
        case COEF:
            grid.min_val = 0.1;
            grid.max_val = 300;
            grid.step = 14; // total iterations = 3
            break;
        case DEGREE:
            grid.min_val = 0.01;
            grid.max_val = 4;
            grid.step = 7; // total iterations = 3
            break;
        default:
            return SVMParamGrid();  // Type is invalid
            break;
        }

        return grid;
    }

#ifdef CV_VERSION
    SVMParamGrid& operator= (const cv::ParamGrid& grid) {
        min_val  = grid.min_val;
        max_val  = grid.max_val;
        step = grid.step;

        return *this;
    }

    operator cv::ParamGrid() const {
        cv::ParamGrid grid(min_val, max_val, step);
        return grid;
    }
#endif

    // The searching parameters is defined as: min, min*step, min*step^2, ... min*step^n
    // where n should satisfy min*step^n < max
    double min_val  = 0;
    double max_val  = 0;
    double step = 0;
};

struct SVMModelDeletor {
    void operator()(SVMModel* ptr) {
        svm_free_and_destroy_model(&ptr);
    }
};


class SVM {
public:
    // SVM type
    enum { C_SVC = 100, NU_SVC = 101, ONE_CLASS = 102, EPS_SVR = 103, NU_SVR = 104 };

    // Kernel type
    enum { LINEAR = 0, POLY = 1, RBF = 2, SIGMOID = 3};

    void train(const cv::Mat& train_data, cv::InputArray labels, const SVMParameter& params);
    void train(const cv::Mat& train_data, cv::InputArray labels,
               int svm_type = C_SVC, int kernel_type = RBF, int k_fold = 10,
               SVMParamGrid Cgrid      = SVMParamGrid::getDefault(SVMParamGrid::C),
               SVMParamGrid gammaGrid  = SVMParamGrid::getDefault(SVMParamGrid::GAMMA),
               SVMParamGrid pGrid      = SVMParamGrid::getDefault(SVMParamGrid::P),
               SVMParamGrid nuGrid     = SVMParamGrid::getDefault(SVMParamGrid::NU),
               SVMParamGrid coeffGrid  = SVMParamGrid::getDefault(SVMParamGrid::COEF),
               SVMParamGrid degreeGrid = SVMParamGrid::getDefault(SVMParamGrid::DEGREE));

    double predict(cv::InputArray sample) const;
    double predict(const SVMNode& sample) const;

    double predict(cv::InputArray sample, vector<double>& vals, bool return_prob_val = false) const;
    double predict(const SVMNode& sample, vector<double>& vals, bool return_prob_val = false) const;

    auto getParams() const -> const SVMParameter& {
        return params;
    }

private:
    cv::Mat train_mat;
    cv::Mat label_mat;
    SVMParameter params;
    unique_ptr<SVMModel, SVMModelDeletor> model_ptr;
};
} // End namespace ml
_CLANY_END

#endif // SVM_H