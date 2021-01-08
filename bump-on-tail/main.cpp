/* =====================================================================================
 *       Filename: main.cpp
 *         Author: Paul P. Hilscher (2019-)
 *        License: GPLv3+ (https://www.gnu.org/licenses/gpl-3.0.en.html)
 * ====================================================================================*/

#include "BumpOnTail.h"

#include<cmath>
#include<iostream>


int main(int argc, char *argv[])
{
  int Nx = 512,
      Nv = 512;

  double Lx = 10., 
         Lv = 12.; 

  double stopTime = 200.;
  int    maxSteps = 1e5;;


  for (int i = 1; i < argc; ++i) 
  {
    if (!strcmp(argv[i], "-S"))
    {
      maxSteps = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "-T"))
    {
      stopTime = atof(argv[++i]);
    }
    else if (!strcmp(argv[i], "-Nx"))
    {
      Nx = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "-Nv"))
    {
      Nv = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "-Lx"))
    {
      Lx = atof(argv[++i]);
    }
    else if (!strcmp(argv[i], "-Lv"))
    {
      Lv = atof(argv[++i]);
    }
    else
    {
      printf( "Bump on tail options:\n" );
      printf( "  -S  <int>  :  number of steps (default: 1e5)\n");
      printf( "  -T  <float>:  time  (default: 100)\n");
      printf( "  -Nx <int>  :  x grid points  (default: 512)\n");
      printf( "  -Nv <int>  :  v grid points  (default: 512)\n");
      printf( "  -Lx <float>:  Lx size (default: 10 unit <2pi>)\n");
      printf( "  -Lv <float>:  Lv size (default: 10)\n");
      exit( 1 );
    }
  }

  ////////////////////////////////////////////////////

  Kokkos::initialize(argc, argv);

  BumpOnTail bot(Nx, Nv, 2.*M_PI*Lx, Lv);

  while((bot.getTime() < stopTime) && (bot.getStep() < maxSteps))
  {
    bot.step();
  }

  bot.write();
  
  std::cout << "Done" << std::endl;
 
  Kokkos :: finalize ();
  // C-99/C++: main implicitly returns EXIT_SUCCESS
}
