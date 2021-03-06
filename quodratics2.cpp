//g++ quodratics2.cpp
//g++ quodratics2.cpp -llapacke -llapack -lcblas
#include<cmath>

#define  CHAINS 50
#define  DT0 0.000000001
#define  decaydt 0.1
#define decayenergy 0.1
#define STEPS 5
#define INTERVAL 1001
#define DIAG false

#include <time.h>
#include <fstream>

#define DIM 2

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

const int BURNIN =  9000;
const int EPISODE = 10000;
double QS[(EPISODE-BURNIN)*CHAINS][DIM];
int idx=0;
void gen(double* q)
{
  QS[idx][0]=q[0];
  QS[idx][1]=q[1];
  idx++;
}

#include "Sampler2.h"

int main()
{
  clock_t start, end;
  double cpu_time_used;
  start = clock();

  hmc(true, true, NULL);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("%f seconds",cpu_time_used);
  std::ofstream csv("./QS.csv");
  for(int i=0;i<(EPISODE-BURNIN)*CHAINS;i++){
    csv<< QS[i][0] << "," << QS[i][1] << std::endl;
  }
  csv.close();
  return 0;
}

