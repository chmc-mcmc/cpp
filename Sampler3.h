//install blas, lapack
//https://blog.csdn.net/mlnotes/article/details/9676269
#include <lapacke.h>
#include <cmath>
#include <string.h>
#include <random>
extern "C" {
#include  <cblas.h>
}
#define  DT0 0.000000001
#define  decaydt 0.1
#define decayenergy 0.1
#define STEPS 5
using namespace std;

extern lapack_int LAPACKE_dgesv( int matrix_order, lapack_int n, lapack_int nrhs,
                                 double * a, lapack_int lda, lapack_int * ipiv,
                                 double * b, lapack_int ldb );

inline double clip(double n, double lower, double upper)
{
  return std::max(lower, std::min(n, upper));
}
class HMC{
 private:
  bool vanilla;
  bool chmc;
  double *qAll;
  double *pAll;
  double Htotal1;
  double Htotal2;
  double Htotal;
  double Utotal;
  double dt1 = DT0;
  double dt2 = DT0;
  double dt;
  double s[CHAINS];
  double S[CHAINS];
  double AS[CHAINS];
  bool switch1;
  std::default_random_engine generator;

  int iter = 0;
  int chain = 0;
 protected:
 public:
  HMC(bool vanilla, bool switch1, const double *qinit)
  {
    this->vanilla = vanilla;
    this->switch1 = switch1;
    qAll= new double[CHAINS*DIM];
    pAll= new double[CHAINS*DIM];
    if(qinit!=NULL)
      memcpy(qAll,qinit,CHAINS*DIM*sizeof(double));
    else{
      std::normal_distribution<double> normal(0.0,1.);
      std::uniform_real_distribution<double> uniform(0.0,1.0);

      for(int i=0; i<CHAINS*DIM;i++)
        qAll[i] = normal(generator);
    }
    Utotal = 0;
    for(int i=0;i<CHAINS;i++)
      Utotal += U(qAll+i*DIM);
    Htotal1 = 2*Utotal;
    Htotal2 = 2*Utotal;
  }
  ~HMC()
  {
    delete [] pAll;
    delete [] qAll;
  }

  double *sample()
  {
    std::normal_distribution<double> normal(0.0,1.);
    std::uniform_real_distribution<double> uniform(0.0,1.0);
    if(chain == 0) {
      Utotal = 0;
      for(int i=0;i<CHAINS;i++)
        Utotal += U(qAll+i*DIM);
      for(int i=0; i<CHAINS*DIM;i++)
        pAll[i] = normal(generator);
      double KtotalNew = 0;
      if(vanilla){
        for(int i=0;i<CHAINS;i++){
          KtotalNew += cblas_ddot(DIM, pAll+i*DIM, 1, pAll+i*DIM, 1)/2;
        }
      }else{
        for(int i=0;i<CHAINS;i++){
          double *q=qAll+i*DIM;
          double *p=pAll+i*DIM;
          double a[DIM][DIM];
          double x[DIM];
          ddU(q, &a[0][0]);
          memcpy(x, p, DIM*sizeof(double));
          if(DIAG){
            double *a1=a[0];
            for(int k=0;k<DIM;k++){
              x[k]/=a1[k];
            }
          }else{
            int ipiv[DIM];
            int n = DIM;
            int nrhs = 1 ;
            int lda = DIM;
            int ldb = DIM;
            int info = LAPACKE_dgesv(LAPACK_COL_MAJOR,n,nrhs,a[0],lda,ipiv,x,ldb);
          }
          KtotalNew += cblas_ddot(DIM, p, 1, x, 1)/2;
        }
      }
      if(vanilla){
        Htotal = Htotal1;
        dt = dt1;
      }else{
        Htotal = Htotal2;
        dt = dt2;
      }
      double Ktotal = Htotal - Utotal;
      double scale = std::sqrt(std::abs(Ktotal/KtotalNew));
      for(int i=0;i<DIM*CHAINS;i++)
        pAll[i]=pAll[i]*scale;
    }
    bool bad = false;
    double *p = pAll + chain * DIM;
    double *q = qAll + chain * DIM;
    double UE[STEPS+1];
    UE[0] = U(q);
    double q0[DIM];
    memcpy(q0,q,sizeof(double)*DIM);
    for(int k=0;k<STEPS;k++){
      double dudq[DIM];
      dU(q,dudq);
      for(int h=0;h<DIM;h++){
        p[h]-=dt*dudq[h];
      }
      double q1[DIM];
      memcpy(q1,q,sizeof(double)*DIM);
      if(vanilla){
        for(int h=0;h<DIM;h++){
          q[h]+=dt*p[h];
        }
      }else{
        double a[DIM][DIM];
        double x[DIM];
        ddU(q, &a[0][0]);
        memcpy(x, p, DIM*sizeof(double));
        if(DIAG){
          double *a1=a[0];
          for(int k=0;k<DIM;k++){
            x[k]/=a1[k];
          }
        }else{
          int ipiv[DIM];
          int n = DIM;
          int nrhs = 1 ;
          int lda = DIM;
          int ldb = DIM;
          int info = LAPACKE_dgesv(LAPACK_COL_MAJOR,n,nrhs,a[0],lda,ipiv,x,ldb);
        }
        for(int h=0;h<DIM;h++){
          q[h]+=dt*x[h];
        }
      }
      if(outbnd!=NULL && outbnd(q)){
        bad = true;
        memcpy(q,q1,sizeof(double)*DIM);
      }
      UE[k+1]=U(q);
    }
    if(outbnd(q)){
      bad = true;
    }
    //14. compute Si and si
    int mini = 0;
    int maxi = 0;
    for(int k=1; k<= STEPS; k++){
      if(UE[k]>UE[maxi])
        maxi = k;
      else if(UE[k]<UE[mini])
        mini = k;
    }
    s[chain] = mini;
    S[chain] = maxi;
    double alpha = 0.0;
    if(!bad){
      alpha = std::exp(clip(U(q0)-U(q),-200.0,0.0));
      //printf("%g %g %g %g\n",Htotal1, Htotal2, dt1, dt2);
    }
    AS[chain]=alpha;
    if(alpha < uniform(generator)){
      memcpy(q,q0,sizeof(double)*DIM);
    }
    memcpy(qAll + chain * DIM, q, sizeof(double)*DIM);
    if(chain == CHAINS-1){
      if(iter<BURNIN){
        double ap=0;
        int s0=0,s1=0,S0=0,S1=0;
        for(int k = 0; k< CHAINS; k++){
          ap += AS[k];
          if(s[k]==0)
            s0++;
          if(s[k]==STEPS)
            s1++;
          if(S[k]==0)
            S0++;
          if(S[k]==STEPS)
            S1++;
        }
        ap /= CHAINS;
        if(s0==CHAINS && S1==CHAINS){
          dt /= (1+decaydt);
        }else if(s0+s1==CHAINS && S0+S1==CHAINS){
          dt *= (1+decaydt);
        }
        if(ap>.9){
          Htotal = (Htotal-Utotal)*(1+decayenergy)+Utotal;
        }else if(ap<0.1){
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
      if(switch1){
        vanilla = !vanilla;
      }
      chain = 0;
      iter++;
    }else{
      chain++;
    }
    return q;
  }
};
