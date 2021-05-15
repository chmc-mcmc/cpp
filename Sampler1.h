void hmc(double (*U)(double*), void (*dU)(double*, double*), void (*ddU)(double*,double*),bool (*outbnd)(double*), void (*gen)(double*),int Dim, int BURNIN, int EPISODE, bool vanilla, bool switch1, const double *qinit);

#define  CHAINS 10
#define  DT0 0.000000001
#define  decaydt 0.1
#define decayenergy 0.1
#define STEPS 20
#define switch0 100
