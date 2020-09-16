include "cyarma.pyx"
cimport numpy as np
import numpy as np
cdef dosomething_x(np.ndarray[np.double_t, ndim=2] X):
  cdef mat *ax=new mat(<double*>X.data,X.shape[0],X.shape[1],False,True)
  cdef double * Xptr = ax.memptr()
  cdef np.ndarray[np.double_t, ndim=2] D = np.empty((ax.n_rows, ax.n_cols), dtype=np.double, order="F")
  cdef double * Dptr = <double*> D.data
  for i in range(ax.n_rows*ax.n_cols):
    Dptr[i] = Xptr[i]
  return D
def dosomething(np.ndarray[np.double_t,ndim=2] X):
  return dosomething_x(X)
#Code is going here
cdef extern from "allfunctions.h":
  void linGradCalc(int* nrow, double *eta, double *y, double *ldot)
  double linNegLogLikelihoodCalc(int *nrow, double *eta, double *y)
  void linSolverLamGrid(double *X, double *y, int* index, int *nrow, int *ncol, int *numGroup, double *beta, int *rangeGroupInd, int *groupLen, double *lambda1, double *lambda2, int *innerIter, double *thresh, double *ldot, double *nullBeta, double *gamma, double *eta, int* betaIsZero, int& groupChange, int* isActive, int* useGroup, double *step, int *reset)
 
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

#Code is ended here