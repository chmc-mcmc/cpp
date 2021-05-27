//install blas, lapack
//https://blog.csdn.net/mlnotes/article/details/9676269
//g++ quodratics2oob.cpp -llapacke -lcblas
#include<cmath>
#include <time.h>
#include <fstream>

#define CHAINS 50
#define DIAG false
#define DIM 2
#define EPISODE 10000
#define BURNIN  9000

const double rho = 0.9999999;
double U(double* q)
{
  double x=q[0];
  double y=q[1];
  return 0.5*(x*x+y*y-2.*x*y*rho)/(1.-rho*rho);
}
void dU(double*q, double*dudq)
{
  double x=q[0];
  double y=q[1];
  dudq[0]=0.5*(2.*x-2.*y*rho)/(1.-rho*rho);
  dudq[1]=0.5*(2.*y-2.*x*rho)/(1.-rho*rho);
}
void ddU(double* q,double* h)
{
  double x=q[0];
  double y=q[1];
  h[0] = 1./(1.-rho*rho);
  h[1] = -rho/(1.-rho*rho);
  h[2] = -rho/(1.-rho*rho);
  h[3] = 1./(1.-rho*rho);
}
bool outbnd(double* q){
  return false;
}

#include "Sampler3.h"

int main()
{
  clock_t start, end;
  double cpu_time_used;
  start = clock();
  HMC hmc(true, true
          , NULL);
  double QS[EPISODE-BURNIN][CHAINS][DIM];
  for(int i=0;i<EPISODE;i++){
    for(int j=0;j<CHAINS;j++){
      double *q=hmc.sample();
      if(i>=BURNIN)
        memcpy(QS[i-BURNIN][j],q,DIM*sizeof(double));
    }
  }

  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("%f seconds",cpu_time_used);
  std::ofstream csv("./QS.csv");
  for(int i=0;i<EPISODE-BURNIN;i++){
    for(int j=0;j<CHAINS;j++){
      csv<< QS[i][j][0] << "," << QS[i][j][1] << std::endl;
    }
  }
  csv.close();
  return 0;
}
