//g++ -I/var/u1/work/NRinC302/code test.cpp
#include<cmath>
//#include<stdio.h>

#include <random>
#include "nr3.h"
#include "gaussj.h"
#include "sort.h"
#include "moment.h"
#include "Sampler1.h"
double dot(const double* x, const double *y, const int n)
{
	double sum=0.0;
	for (int i=0;i<n;i++)
		sum += x[i]*y[i];
	return sum;
}
inline double clip(double n, double lower, double upper)
{
  return std::max(lower, std::min(n, upper));
}

// const int CHAINS = 10;
// const double DT0 = 0.000000001;
// const double decaydt = 0.1;
// const double decayenergy = 0.1;
// const int STEPS =20;


void hmc(double (*U)(double*), void (*dU)(double*, double*), void (*ddU)(double*,double*),bool (*outbnd)(double*), void (*gen)(double*),int Dim, int BURNIN, int EPISODE, bool vanilla, bool switch1, const double *qinit)
{
  std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.);
  std::uniform_real_distribution<double> uni(0.0,1.0);

  double *qAll= new double[CHAINS*Dim];
  double *pAll= new double[CHAINS*Dim];
  if(qinit!=NULL)
    memcpy(qAll,qinit,CHAINS*Dim*sizeof(double));
   else
     for(int i=0; i<CHAINS*Dim;i++)
       qAll[i] = distribution(generator);
  double Utotal = 0;
  for(int i=0;i<CHAINS;i++)
    Utotal += U(qAll+i*Dim);
  double Htotal1 = 2*Utotal;
  double Htotal2 = 2*Utotal;
  double dt1 = DT0;
  double dt2 = DT0;
  for(int j=0; j<EPISODE;j++){
    for(int i=0; i<CHAINS*Dim;i++)
      pAll[i] = distribution(generator);
    double KtotalNew = 0;
    if(vanilla){
      for(int i=0;i<CHAINS;i++){
        KtotalNew += dot(pAll+i*Dim, pAll+i*Dim, Dim)/2.;
      }
    }else{
      for(int i=0;i<CHAINS;i++){
        double *q=qAll+i*Dim;
        double *p=pAll+i*Dim;
        MatDoub a(Dim,Dim);
        ddU(q,&a[0][0]);
        MatDoub b(Dim,1);
        memcpy(&b[0][0],p, sizeof(double)*Dim);
        gaussj(a,b);
        KtotalNew += dot(p, &b[0][0], Dim)/2.;
      }
    }

    double Utotal = 0;
    for(int i=0;i<CHAINS;i++)
      Utotal += U(qAll+i*Dim);

    double Htotal, dt;
    if(vanilla){
      Htotal = Htotal1;
      dt = dt1;
    }else{
      Htotal = Htotal2;
      dt = dt2;
    }
    //if(j && j>4990&&j<5010)
    //      printf("%f\n",Htotal);
    double Ktotal = Htotal - Utotal;
    double scale = std::sqrt(std::abs(Ktotal/KtotalNew));
    for(int i=0;i<Dim*CHAINS;i++)
      pAll[i]=pAll[i]*scale;
    int S[2]={0,0};
    int s[2]={0,0};
    VecDoub AS(CHAINS);

    for(int i =0;i<CHAINS;i++){
      bool bad = false;
      double *p = pAll+i*Dim;
      double *q = qAll+i*Dim;
      double UE[STEPS+1];
      UE[0] = U(q);
      double q0[Dim];
      memcpy(q0,q,sizeof(double)*Dim);


      for(int k=0;k<STEPS;k++){
        double dudq[Dim];
        dU(q,dudq);
        for(int h=0;h<Dim;h++){
          p[h]-=dt*dudq[h];
        }
        double q1[Dim];
        memcpy(q1,q,sizeof(double)*Dim);
        if(vanilla){
          for(int h=0;h<Dim;h++){
            q[h]+=dt*p[h];
          }
        }else{
          MatDoub a(Dim,Dim);
          ddU(q,&a[0][0]);
          MatDoub b(Dim,1);
          memcpy(&b[0][0],p, sizeof(double)*Dim);
          gaussj(a,b);
           for(int h=0;h<Dim;h++){
            q[h]+=dt*b[h][0];
          }
        }
        if(outbnd!=NULL && outbnd(q)){
          memcpy(q,q1,sizeof(double)*Dim);
          bad = true;
        }
        UE[k+1]=U(q);
      }


      double alpha = 0.0;
      if(!bad){
        alpha = std::exp(clip(U(q0)-U(q),-200.0,0.0));
      }
      AS[i]=alpha;
      if(alpha < uni(generator)){
        memcpy(q,q0,sizeof(double)*Dim);
      }
      memcpy(qAll+i*Dim,q,sizeof(double)*Dim);

      if(j>=BURNIN){
        gen(q);
      }
      //else{
        NRvector<Doub> UE1(STEPS+1,UE);
        NRvector<Int> r(STEPS+1);
        Indexx idx(UE1);
        idx.rank(r);
        if(r[0]==0){
          s[0]++;
        }else if(r[STEPS]==0){
          s[1]++;
        }
        if(r[0]==STEPS){
          S[0]++;
        }else if(r[STEPS]==STEPS){
          S[1]++;
        }
        //}
    }
    Doub ave, var;
    avevar(AS, ave,  var);
    if(j<BURNIN){
      if(s[0]==CHAINS && S[1]==CHAINS){
        dt /= (1+decaydt);
      }else if(s[0]+s[1]==CHAINS && S[0]+S[1]==CHAINS){
        dt *= (1+decaydt);
      }
      if(ave>.9){
        Htotal = (Htotal-Utotal)*(1+decayenergy)+Utotal;
      }else if(ave<0.1){
        Htotal = (Htotal-Utotal)/(1+decayenergy)+Utotal;
      }
      if(vanilla){
        Htotal1 = Htotal;
        dt1 = dt;
      }else{
        Htotal2 = Htotal;
        dt2 = dt;
      }
    }
    if(j==0)
      std::cout<<setw(5)<<"j"
               <<setw(10)<<"Ktotal"
               <<setw(10)<<"KtotalNew"
               <<setw(10)<<"Utotal"
               <<setw(10)<<"Htotal1"
               <<setw(10)<<"Htotal2"
               <<setw(12)<<"dt1"
               <<setw(12)<<"dt2"
               <<setw(5)<<"va"
               <<setw(10)<<"mean"
               <<setw(5)<<"s0"
               <<setw(5)<<"s1"
               <<setw(5)<<"S0"
               <<setw(5)<<"S1"<<endl;
    else if(j % 11==0)
    //if(j && j>4990&&j<5010)
      //printf("%d\t: %f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%f\t%d %d %d %d\n",j, Ktotal,KtotalNew,Utotal,Htotal,Htotal1,Htotal2,dt1,dt2,vanilla,ave,s[0],s[1],S[0],S[1]);
      std::cout<<setw(5)<<j
               <<setw(10)<<Ktotal
               <<setw(10)<<KtotalNew
               <<setw(10)<<Utotal
               <<setw(10)<<Htotal1
               <<setw(10)<<Htotal2
               <<setw(12)<<dt1
               <<setw(12)<<dt2
               <<setw(5)<<vanilla
               <<setw(10)<<ave
               <<setw(5)<<s[0]
               <<setw(5)<<s[1]
               <<setw(5)<<S[0]
               <<setw(5)<<S[1]<<endl;
    if(switch1){
      vanilla = !vanilla;
    }
  }
  delete [] pAll;
  delete [] qAll;
}
// const double rho = 0.99999999;
// double U(double* q)
// {
//   double x=q[0];
//   double y=q[1];
//   return 0.5*(x*x+y*y-2.*x*y*rho)/(1.-rho*rho);
// }
// void dU(double*q, double*dudq)
// {
//   double x=q[0];
//   double y=q[1];
//   dudq[0]=0.5*(2.*x-2.*y*rho)/(1.-rho*rho);
//   dudq[1]=0.5*(2.*y-2.*x*rho)/(1.-rho*rho);
// }
// void ddU(double* q,double* h)
// {
//   double x=q[0];
//   double y=q[1];
//   h[0] = 1./(1.-rho*rho);
//   h[1] = -rho/(1.-rho*rho);
//   h[2] = -rho/(1.-rho*rho);
//   h[3] = 1./(1.-rho*rho);
// }

// const int EPISODE = 10000;
// const int BURNIN =  5000;
// const int Dim = 2;
// const int INTERVAL=1001;
// double QS[(EPISODE-BURNIN)*CHAINS][Dim];
// bool outbnd(double* q){
//   return false;
// }
// int idx=0;
// void gen(double* q)
// {
//   QS[idx][0]=q[0];
//   QS[idx][1]=q[1];
//   idx++;
// }
// int main()
// {
//   hmc(U,dU,ddU,outbnd,gen,Dim, BURNIN, EPISODE, true, true, NULL);
//    ofstream csv("./QS.csv");
//    for(int i=0;i<(EPISODE-BURNIN)*CHAINS;i++){
//      csv<<QS[i][0]<<","<<QS[i][1]<<endl;
//    }
//    csv.close();

//   return 0;
// }

