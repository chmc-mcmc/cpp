//g++ -I~/work/NRinC302/code quodratics2.cpp Sampler1.cpp
#include<cmath>
#include "Sampler1.h"
#include <fstream>

const double rho = 0.99999999;
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

const int EPISODE = 10000;
const int BURNIN =  5000;
const int Dim = 2;
const int INTERVAL=1001;

double QS[(EPISODE-BURNIN)*CHAINS][Dim];
bool outbnd(double* q){
  return false;
}

int idx=0;
void gen(double* q)
{
  QS[idx][0]=q[0];
  QS[idx][1]=q[1];
  idx++;
}

int main()
{
  hmc(U,dU,ddU,outbnd,gen,Dim, BURNIN, EPISODE, true, true, NULL);
   std::ofstream csv("./QS.csv");
   for(int i=0;i<(EPISODE-BURNIN)*CHAINS;i++){
     csv<< QS[i][0] << "," << QS[i][1] << std::endl;
   }
   csv.close();

  return 0;
}

