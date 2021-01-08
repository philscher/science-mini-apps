/* =====================================================================================
 *       Filename: ndarray.h
 *         Author: Paul P. Hilscher (2019-)
 *        License: GPLv3+ (https://www.gnu.org/licenses/gpl-3.0.en.html)
 * ====================================================================================*/

#pragma once

#include <Kokkos_Core.hpp>


struct Array1d
{
  double *_data; 

  Array1d(int Nx) : _Nx(Nx)
  { 
    _data = (double *) Kokkos::kokkos_malloc<Kokkos::CudaUVMSpace>(sizeof(double) * Nx);
  }
 
  KOKKOS_FUNCTION inline double& operator()(int x)
  {
    return _data[x];
  }

  KOKKOS_FUNCTION inline double  operator()(int x) const
  {
    return _data[x];
  }

  unsigned _Nx;
};


struct Array2d
{
  double *_data; 

  Array2d(int Nx, int Ny) : _Nx(Nx)
      			  , _Ny(Ny)
  { 
    _data = (double *) Kokkos::kokkos_malloc<Kokkos::CudaUVMSpace>(sizeof(double) * Nx * Ny);
  }
  
  KOKKOS_FUNCTION inline double& operator()(int x, int y)
  {
    return _data[x*_Ny+y];
  }

  KOKKOS_FUNCTION inline double  operator()(int x, int y) const
  {
    return _data[x*_Ny+y];
  }

  unsigned _Nx;
  unsigned _Ny;
};
