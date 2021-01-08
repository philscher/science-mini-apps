/* =====================================================================================
 *       Filename: BumpOnTail.cpp
 *         Author: Paul P. Hilscher (2019-)
 *        License: GPLv3+ (https://www.gnu.org/licenses/gpl-3.0.en.html)
 * ====================================================================================*/

#include "BumpOnTail.h"


#include<complex>
#include<iomanip>
#include<iostream>
#include<fstream>


namespace
{
  template<typename T>
  inline T pow2(T const &t)
  {
    return t*t;
  }
}


BumpOnTail::BumpOnTail(int const Nx, int const Nv, double const Lx, double const Lv)
 : _Nx(Nx)
 , _Nv(Nx)
 , _Lx(Lx)
 , _Lv(Lv)
 , _step(0)
 , _time(0)
 , _nghost(2)
 , _V(int(Nv))
 , _X(int(Nx))
 , _k2(int(Nx))
 , _phi(int(Nx))
 , _f(Nx, Nv)
 , _g(Nx, Nv)
 , _k(Nx, Nv)
 , _a(Nx, Nv)
{
  init();
}


KOKKOS_FUNCTION
void BumpOnTail::
poisson_accum(int x, Array2d const f, Array1d rho) const
{
  rho(x) = 0;

  for(int v = 0; v < _Nv; ++v)
  { 
    rho(x) += f(x,v);
  } 
      
  rho(x) *= _dv;
  rho(x) -= 1.;
}



void BumpOnTail::
poisson_solve()
{
  std::complex<double> rhok[_Nx-4]; // faster and lighter than alloca

  fftw_execute_dft_r2c(plan_forward, &_phi(2), (fftw_complex *) rhok);

  for(int k=0; k < (_Nx-4)/2+1; ++k)
  {
    rhok[k] *= -_k2(k) ;
  }
  
  fftw_execute_dft_c2r(plan_backward, (fftw_complex *) rhok, &_phi(2));
  poisson_bounds();
}


KOKKOS_FUNCTION
void BumpOnTail::
vlasov_evolve(int x, int v, Array2d const f, Array2d g, Array2d a, double const rk[2]) const
{
  const double dphi_dx = (8.*(_phi(x+1) - _phi(x-1)) 
                           - (_phi(x+2) - _phi(x-2))) * _rp_12_dx; 
           
  const double df_dv = (8.*(f(x,v+1) - f(x,v-1)) 
                         - (f(x,v+2) - f(x,v-2))) * _rp_12_dv;

  const double df_dx = (8. *(f(x+1,v) - f(x-1,v))  
                          - (f(x+2,v) - f(x-2,v))) * _rp_12_dx ;
           
 
  const double df_dt = -_V(v) * df_dx - dphi_dx * df_dv;
    
  // time-integrate distribution function and accumulate immediately
  g(x,v)  = _f(x,v) + rk[0] * df_dt;
  a(x,v) += df_dt * rk[1];
}


KOKKOS_FUNCTION
void BumpOnTail::
vlasov_bounds(int v, Array2d f) const
{
 unsigned const Mx = _Nx - 2*_nghost;
   
  for(int n = 0; n < 2; ++n)
  { 
   f(n,v)       = f(n+Mx,v);
   f(_Nx-n-1,v) = f(_Nx - n - 1 - Mx,v);
  }
}



void BumpOnTail::
poisson_bounds()
{
  unsigned const Mx = _Nx - 2*_nghost;

  for(int n = 0; n < 2; ++n)
  { 
   _phi(n      ) = _phi(n+Mx);
   _phi(_Nx-n-1) = _phi(_Nx - n - 1 - Mx);
  }
} 



KOKKOS_FUNCTION
void BumpOnTail::
vlasov_update(int x, int v, Array2d f, Array2d a) const
{ 
  f(x,v) += _a(x,v) * _dt_rp6;
  a(x,v)  = 0.;  
}

void BumpOnTail::step()
{
  using MDPolicyType_2D = typename Kokkos::Experimental::MDRangePolicy<Kokkos::Experimental::Rank<2>>;

  MDPolicyType_2D vlasov_xv({ 2, 2}, { _Nx-2, _Nv-2});

  Kokkos::RangePolicy<> poisson_x(2,_Nx-2);
  Kokkos::RangePolicy<> vlasov_v (2,_Nv-2);

  // RK Step-1
  const double rk1[2] = { 0.5*_dt , 1 };
  Kokkos::parallel_for("poisson_accum_1", poisson_x, KOKKOS_CLASS_LAMBDA(int x){ poisson_accum(x, _f, _phi); });
  cudaDeviceSynchronize();
  poisson_solve();
  Kokkos::parallel_for("vlasov_evolve_1", vlasov_xv, KOKKOS_CLASS_LAMBDA(int x, int v) { vlasov_evolve(x, v, _f, _g, _a, rk1);});
  cudaDeviceSynchronize();
  Kokkos::parallel_for("vlasov_bounds_1", vlasov_v , KOKKOS_CLASS_LAMBDA(int v) { vlasov_bounds(v, _g);});
  cudaDeviceSynchronize();
  
  // RK Step-2
  const double rk2[2] = { 0.5*_dt , 2. };
  
  Kokkos::parallel_for("poisson_accum_2", poisson_x, KOKKOS_CLASS_LAMBDA(int x){ poisson_accum(x, _g, _phi); });
  Kokkos::fence();
  poisson_solve();
  Kokkos::parallel_for("vlasov_evolve_2", vlasov_xv, KOKKOS_CLASS_LAMBDA(int x, int v){ vlasov_evolve(x, v, _g, _k, _a, rk2);});
  Kokkos::fence();
  Kokkos::parallel_for("vlasov_bounds_2", vlasov_v , KOKKOS_CLASS_LAMBDA(int v) { vlasov_bounds(v, _k);});
  Kokkos::fence();

  // RK Step-3
  const double rk3[2] = { _dt , 2. };
  
  Kokkos::parallel_for("poisson_accum_3", poisson_x, KOKKOS_CLASS_LAMBDA(int x){ poisson_accum(x, _k, _phi); });
  Kokkos::fence();
  poisson_solve();
  Kokkos::parallel_for("vlasov_evolve_3", vlasov_xv, KOKKOS_CLASS_LAMBDA(int x, int v){ vlasov_evolve(x, v, _k, _g, _a, rk3);});
  Kokkos::fence();
  Kokkos::parallel_for("vlasov_bounds_3", vlasov_v , KOKKOS_CLASS_LAMBDA(int v) { vlasov_bounds(v, _g);});
  Kokkos::fence();
  
  // RK Step-4
  const double rk4[2] = { 0. , 1. };
  
  Kokkos::parallel_for("poisson_accum_4", poisson_x, KOKKOS_CLASS_LAMBDA(int x){ poisson_accum(x, _g, _phi); });
  Kokkos::fence();
  poisson_solve();
  Kokkos::parallel_for("vlasov_evolve_4"     , vlasov_xv, KOKKOS_CLASS_LAMBDA(int x, int v){ vlasov_evolve(x, v, _g, _k, _a, rk4);});
  Kokkos::fence();
  Kokkos::parallel_for("vlasov_update"       , vlasov_xv, KOKKOS_CLASS_LAMBDA(int x, int v) { vlasov_update(x,v, _f, _a);});
  Kokkos::fence();
  Kokkos::parallel_for("vlasov_bounds_4"     , vlasov_v , KOKKOS_CLASS_LAMBDA(int v) { vlasov_bounds(v, _f);});
  Kokkos::fence();
  // RK Step - Done
  
  _step += 1;
  _time += _dt;

  if(_step % 100 == 0)
  {
    std::cout << "Step: " << _step << " Time: " << _time << " " << std::setprecision(15) <<  _f(100,80) << std::endl;
  }
 }

void BumpOnTail::
write()
{
  std::ofstream out_file;
  
  out_file.open ("out.dat", std::fstream::out);

  out_file << _Nx-4 << "," << _Nv-4 << "," << _Lx << "," << _Lv << "," << _step << "," << _time << std::endl;
  for(int x = 2; x < _Nx-2; ++x)
  { 
    for(int v = 2; v < _Nv-3; ++v)
    { 
      out_file << _f(x,v) << ",";
    }
    out_file << _f(x,_Nv-3) << std::endl;
  }
  out_file.close();
}


void BumpOnTail::init()
{
  _dv = 2. * _Lv / (_Nv - 1);
  _dx =      _Lx / (_Nx - 1);
  _dt = 0.45 * _dx/_Lv; // CFL condition estimate

  _rp_12_dx = 1./(12.*_dx);
  _rp_12_dv = 1./(12.*_dv);
  _dt_rp6   = _dt / 6.;

  for(int v = 0; v < _Nv; v++) { 
    _V(v) = -_Lv + v * _dv;
  }
  
  for(int x = 2; x < _Nx-2; x++) { 
    _X(x) = (x-2) * _dx;
  }
  
  unsigned const Mx = _Nx - 2*_nghost;
  for(int n = 0; n < 2; ++n)
  { 
   _X(n      ) = _X(n+Mx);
   _X(_Nx-n-1) = _X(_Nx - n - 1 - Mx);
  }

  const double k0 = 2.*M_PI/_Lx;
  
  for(int k=0; k < _Nx-4; ++k)
  {
    _k2(k) = (k == 0) ? 0. : 1./pow2(k0*k)/(_Nx-4.) ;
  }


  double const n1 = 0.9,
	       n2 = 0.2,
	       vt = 0.5,
	       eps = 0.04,
	       v0 = 4.5,
	       n  = 3.;
	       
  double const k = 2*M_PI*(n/_Lx);
  

  // initialization field-perturbation
  // todo: move away from stack
  double f_norm[_Nx];

  for(int x = 0; x < _Nx; ++x) {

    f_norm[x] = 0.;

  for(int v = 0; v < _Nv; ++v) {

    double f = (1./(2.*M_PI)) * (n1 * std::exp(-0.5 * pow2(_V(v))) + n2 * std::exp(-0.5*pow2(_V(v)-v0)/vt));
    _f(x,v) =  f * (1. + eps * std::cos(k * _X(x)));
    _a(x,v) = 0.;
    f_norm[x] += f * _dv;
  } }

  for(int x = 0; x < _Nx; ++x) { for(int v = 0; v < _Nv; ++v) {
    _f(x,v) *= 1./f_norm[x];
   } }

  // todo: move away from stack, memory might not be aligned.
  std::complex<double> *rhok = (std::complex<double> *) alloca( sizeof(std::complex<double>) * (_Nx-4));
  plan_forward = fftw_plan_dft_r2c_1d(_Nx-4, &_phi(2), (fftw_complex *) rhok, FFTW_ESTIMATE);
  plan_backward = fftw_plan_dft_c2r_1d(_Nx-4, ( fftw_complex *) rhok, &_phi(2), FFTW_ESTIMATE);
}
