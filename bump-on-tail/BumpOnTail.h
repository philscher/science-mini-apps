/* =====================================================================================
 *       Filename: BumpOnTail.h
 *         Author: Paul P. Hilscher (2019-)
 *        License: GPLv3+ (https://www.gnu.org/licenses/gpl-3.0.en.html)
 * ====================================================================================*/


#pragma once

#include <Kokkos_Core.hpp>

#include "ndarray.h"

#include<fftw3.h>


class BumpOnTail
{
public:

  BumpOnTail(int const Nx, int const Nv, double const Lx, double const Lv);

  void step();
 
  void write();

  double getTime() const { return _time; }
  double getStep() const { return _step; }

private:

  void init();

  KOKKOS_FUNCTION void vlasov_evolve(int x, int v, Array2d const f, Array2d g, Array2d a, double const rk[2]) const;
  KOKKOS_FUNCTION void vlasov_bounds(int v, Array2d f) const;
  KOKKOS_FUNCTION void vlasov_update(int x, int v, Array2d f, Array2d a) const;

  KOKKOS_FUNCTION void poisson_accum(int x, Array2d const f, Array1d k2) const;

  void poisson_solve();
  void poisson_bounds();

  void rk_step(Array2d in, Array2d out, const double rk[2]);
 
  int const _Nx,
            _Nv;

  double const _Lx,
	       _Lv;
  
  unsigned _step;
  double   _time;
  
  int const  _nghost;
  
  double _dx,
	 _dv,
	 _dt;

  double _rp_12_dx,
         _rp_12_dv,
         _dt_rp6;

  Array1d _V  ;
  Array1d _X  ;
  Array1d _k2 ;
  Array1d _phi;

  Array2d _f;  
  Array2d _g;
  Array2d _k;
  Array2d _a;

  fftw_plan plan_forward;
  fftw_plan plan_backward;
};
