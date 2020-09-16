include "cyarma.pyx"
cimport numpy as np
import numpy as np
#Code is going here
#External functions from allfunctions.h
cdef extern from "allfunctions.h":
  void linGradCalc(int* nrow, double *eta, double *y, double *ldot)
  double linNegLogLikelihoodCalc(int *nrow, double *eta, double *y)
  void linSolverLamGrid(double *X, double *y, int* index, int *nrow, int *ncol, int *numGroup, double *beta, int *rangeGroupInd, int *groupLen, double *lambda1, double *lambda2, int *innerIter, double *thresh, double *ldot, double *nullBeta, double *gamma, double *eta, int* betaIsZero, int& groupChange, int* isActive, int* useGroup, double *step, int *reset)
  void linNestLamGrid(double *X, double*y, int* index, int *nrow, int *ncol,
                    int *numGroup, int *rangeGroupInd, int *groupLen,
                    double *lambda1, double *lambda2, double *beta,
                    int *innerIter, int *outerIter, double *thresh,
                    double *outerThresh, double *eta, double *gamma,
                    int *betaIsZero, double *step, int *reset)
  vec cvfolds(double nfolds, double nrow)
  double getmin_cpp(vec lambda_,vec cvm,vec cvsd, int which_lambda);
  mat cpp_sgl_fit(vec& beta0, mat& Z, mat& X, vec& y, vec& index,
                      vec& lambda1, vec& lambda2,
                      int innerIter, int outerIter, double thresh,
                      double outerThresh, double gamma_solver,
                      double step, int reset)
  mat cpp_sgl_fitpath(mat& X, mat& Z, vec& y, vec& index, double dummies,
                          vec l1_frac, vec l21_frac, vec dummies_index,
                          vec& lambdas, double gamma_w,
                          int innerIter, int outerIter, double thresh,
                          double outerThresh,double gamma_solver,
                          double step, int reset)
  vec fastols(const vec & Y, const mat & X, double intercept)
  vec fastals(const vec & Y, const mat & X, double intercept, double tau, double maxIter, double thresh)
  mat boundc(colvec x, colvec dx) #not wrapped in python some api issue
  vec fastrq(const vec & Y, const mat & X, double intercept, double tau)
  vec midas_pr(const vec & Y, const mat & X, double intercept, double tau, double which_loss, double num_evals)
  vec midasar_pr(const vec & Y, const mat & YLAG, const mat & X, double intercept, double tau, double which_loss, double num_evals)
#Building wrapper for external functions with support of numpy
def lin_grad_calc(np.ndarray[np.int_t,ndim=1] nrow,np.ndarray[np.double_t,ndim=1] eta,np.ndarray[np.double_t,ndim=1] y,np.ndarray[np.double_t,ndim=1] ldot):
  linGradCalc(<int*>nrow.data,<double*>eta.data,<double*>y.data,<double*>ldot.data)

def lin_neg_log_likelihood_calc(np.ndarray[np.int_t,ndim=1] nrow,np.ndarray[np.double_t,ndim=1] eta,np.ndarray[np.double_t,ndim=1] y):
  return linNegLogLikelihoodCalc(<int*>nrow.data,<double*>eta.data,<double*>y.data)

def lin_solver_lam_grid(
        np.ndarray[np.double_t,ndim=1] X,np.ndarray[np.double_t,ndim=1] y,
        np.ndarray[np.int_t,ndim=1] index,np.ndarray[np.int_t,ndim=1] nrow,
        np.ndarray[np.int_t,ndim=1] ncol,np.ndarray[np.int_t,ndim=1] numGroup,
        np.ndarray[np.double_t,ndim=1] beta,np.ndarray[np.int_t,ndim=1] rangeGroupInd,
        np.ndarray[np.int_t,ndim=1] groupLen,np.ndarray[np.double_t,ndim=1] lambda1,np.ndarray[np.double_t,ndim=1] lambda2,
        np.ndarray[np.int_t,ndim=1] innerIter,np.ndarray[np.double_t,ndim=1] thresh,np.ndarray[np.double_t,ndim=1] ldot,
        np.ndarray[np.double_t,ndim=1] nullBeta,np.ndarray[np.double_t,ndim=1] gamma,
        np.ndarray[np.double_t,ndim=1] eta,np.ndarray[np.int_t,ndim=1] betaIsZero,groupChange,np.ndarray[np.int_t,ndim=1] isActive,
        np.ndarray[np.int_t,ndim=1] useGroup,np.ndarray[np.double_t,ndim=1] step,np.ndarray[np.int_t,ndim=1] reset
):
    linSolverLamGrid(
        <double*>X.data,<double*>y.data,<int*>index.data,
        <int*>nrow.data,<int*>ncol.data,<int*>numGroup.data,
        <double*>beta.data,<int*>rangeGroupInd.data,<int*>groupLen.data,
        <double*>lambda1.data,<double*>lambda2.data,<int*>innerIter.data,
        <double*>thresh.data,<double*>ldot.data,<double*>nullBeta.data,
        <double*>gamma.data,<double*>eta.data,<int*>betaIsZero.data,
        groupChange,<int*>isActive.data,<int*>useGroup.data,<double*>step.data,
        <int*>reset.data
    )

def lin_nest_lam_grid(
   np.ndarray[np.double_t,ndim=1] X,np.ndarray[np.double_t,ndim=1] y,
   np.ndarray[np.int_t,ndim=1] index,np.ndarray[np.int_t,ndim=1] nrow,
   np.ndarray[np.int_t,ndim=1] ncol,np.ndarray[np.int_t,ndim=1] numGroup,
   np.ndarray[np.int_t,ndim=1] rangeGroupInd,np.ndarray[np.int_t,ndim=1] groupLen,
   np.ndarray[np.double_t,ndim=1] lambda1,np.ndarray[np.double_t,ndim=1] lambda2,
   np.ndarray[np.double_t,ndim=1] beta,np.ndarray[np.int_t,ndim=1] innerIter,
   np.ndarray[np.int_t,ndim=1] outerIter,np.ndarray[np.double_t,ndim=1] thresh,
   np.ndarray[np.double_t,ndim=1] outerThresh,np.ndarray[np.double_t,ndim=1] eta,
   np.ndarray[np.double_t,ndim=1] gamma,np.ndarray[np.int_t,ndim=1] betaIsZero,
   np.ndarray[np.double_t,ndim=1] step,np.ndarray[np.int_t,ndim=1] reset
):
  linNestLamGrid(
      <double*>X.data,<double*>y.data,
      <int*>index.data,<int*>nrow.data,
      <int*>ncol.data,<int*>numGroup.data,
      <int*>rangeGroupInd.data,<int*>groupLen.data,
      <double*>lambda1.data,<double*>lambda2.data,
      <double*>beta.data,<int*>innerIter.data,
      <int*>outerIter.data,<double*>thresh.data,
      <double*>outerThresh.data,<double*>eta.data,
      <double*>gamma.data,<int*>betaIsZero.data,
      <double*>step.data,<int*>reset.data
  )
def cvfold_s(nfolds,nrow):
    cdef vec result=cvfolds(nfolds,nrow)
    cdef double* ptr=result.memptr()
    return vec_to_numpy(result,None)
def getmin(np.ndarray[np.double_t,ndim=1] lambda_,np.ndarray[np.double_t,ndim=1] cvm,np.ndarray[np.double_t,ndim=1] cvsd,which_lambda):
    return getmin_cpp(numpy_to_vec(lambda_)[0],<vec>numpy_to_vec(cvm)[0],<vec>numpy_to_vec(cvsd)[0],which_lambda)
def sgl_fit(
        np.ndarray[np.double_t,ndim=1] beta0,np.ndarray[np.double_t,ndim=2] Z,
        np.ndarray[np.double_t,ndim=2] X,np.ndarray[np.double_t,ndim=1] y,
        np.ndarray[np.double_t,ndim=1] index,np.ndarray[np.double_t,ndim=1] lambda1,
        np.ndarray[np.double_t,ndim=1] lambda2,innerIter,outerIter,thresh,
        outerThresh,gamma_solver,step,reset
):
  return mat_to_numpy(cpp_sgl_fit(
    numpy_to_vec(beta0)[0],
    numpy_to_mat(Z)[0],
    numpy_to_mat(X)[0],
    numpy_to_vec(y)[0],
    numpy_to_vec(index)[0],
    numpy_to_vec(lambda1)[0],
    numpy_to_vec(lambda2)[0],
    innerIter,outerIter,thresh,
    outerThresh,gamma_solver,step,reset),None)
def sgl_fitpath(
    np.ndarray[np.double_t,ndim=2] X,np.ndarray[np.double_t,ndim=2] Z,
    np.ndarray[np.double_t,ndim=1] y,np.ndarray[np.double_t,ndim=1] index,dummies,
    np.ndarray[np.double_t,ndim=1] l1_frac,np.ndarray[np.double_t,ndim=1] l21_frac,
    np.ndarray[np.double_t,ndim=1] dummies_index,np.ndarray[np.double_t,ndim=1] lambdas,
    gamma_w,innerIter,outerIter,thresh,outerThresh,gamma_solver,step,reset
):
    return mat_to_numpy(
      cpp_sgl_fitpath(
          numpy_to_mat(X)[0],
          numpy_to_mat(Z)[0],
          numpy_to_vec(y)[0],
          numpy_to_vec(index)[0],
          dummies,
          numpy_to_vec(l1_frac)[0],
          numpy_to_vec(l21_frac)[0],
          numpy_to_vec(dummies_index)[0],
          numpy_to_vec(lambdas)[0],
          gamma_w,innerIter,outerIter,thresh,outerThresh,gamma_solver,step,reset
      )
    ,None)
def fostol_s(
        np.ndarray[np.double_t,ndim=1] Y,
        np.ndarray[np.double_t,ndim=2] X,
        intercept
):
  return vec_to_numpy(fastols(numpy_to_vec(Y)[0],numpy_to_mat(X)[0],intercept),None)
def fastal_s(
        np.ndarray[np.double_t,ndim=1] Y,
        np.ndarray[np.double_t,ndim=2] X,
        intercept,tau,maxIter,thresh
):
  return vec_to_numpy(
    fastals(
        numpy_to_vec(Y)[0],
        numpy_to_mat(X)[0],
        intercept,tau,maxIter,thresh
    )
  ,None)
def fastr_q(
        np.ndarray[np.double_t,ndim=1] Y,
        np.ndarray[np.double_t,ndim=2] X,
        intercept,tau
):
  return vec_to_numpy(
    fastrq(
        numpy_to_vec(Y)[0],
        numpy_to_mat(X)[0],
        intercept,tau
    )
  ,None)
def midaspr(
        np.ndarray[np.double_t,ndim=1] Y,
        np.ndarray[np.double_t,ndim=2] X,
        intercept,tau,which_loss,num_evals
):
  return vec_to_numpy(
    midas_pr(
        numpy_to_vec(Y)[0],
        numpy_to_mat(X)[0],
        intercept,tau,which_loss,num_evals
    )
  ,None)
def midasarpr(
        np.ndarray[np.double_t,ndim=1] Y,
        np.ndarray[np.double_t,ndim=2] YLAG,
        np.ndarray[np.double_t,ndim=2] X,
        intercept,tau,which_loss,num_evals
):
    return vec_to_numpy(
        midasar_pr(
            numpy_to_vec(Y)[0],
            numpy_to_mat(YLAG)[0],
            numpy_to_mat(X)[0],
            intercept,tau,which_loss,num_evals
        ),
        None)
#Code is ended here