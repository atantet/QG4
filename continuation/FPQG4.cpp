#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODECont.hpp>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file simQG4.cpp 
 *  \ingroup examples
 *  \brief Simulate quasi-geostrophic double gyre model.
 *
 *  Simulate quasi-geostrophic double gyre model
 *  truncated to 4 modes (Simmonnet, Ghil, Dijkstra, 2005).
 */


/* *  \brief Simulate quasi-geostrophic double gyre model.
 *
 *  Simulate quasi-geostrophic double gyre model
 *  truncated to 4 modes (Simmonnet, Ghil, Dijkstra, 2005).
 *  After parsing the configuration file,
 *  the vector field of the QG4 flow and the Runge-Kutta numerical scheme of order 4 are defined.
 *  The model is then integrated forward and the results saved.
 */
int main(int argc, char * argv[])
{
  // Read configuration file
  if (argc < 2)
    {
      std::cout << "Enter path to configuration file:" << std::endl;
      std::cin >> configFileName;
    }
  else
    {
      strcpy(configFileName, argv[1]);
    }
  try
   {
     readConfig(configFileName);
    }
  catch (...)
    {
      std::cerr << "Error reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }

  gsl_vector *x0 = gsl_vector_alloc(dim);
  gsl_vector *sol = gsl_vector_alloc(dim);
  gsl_matrix *solJac = gsl_matrix_alloc(dim, dim);
  const double eps = 1.e-6;
  const int maxIter = 1000;

  // Initialize state and fundamental matrix
  gsl_vector_set_zero(x0);
  gsl_vector_set(x0, 1, 0.473693);
  
  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new QG4(sigma, ci, li);
  
  // Define linearized field
  std::cout << "Defining Jacobian, initialized at x0..." << std::endl;
  linearField *Jacobian = new JacobianQG4(sigma, ci, li, x0);

  // Define fixed point problem
  fixedPointCorr *track = new fixedPointCorr(field, Jacobian, eps, eps, maxIter);

  // Find fixed point
  track->findSolution(x0);

  if (!track->hasConverged())
    std::cerr << "Did not converge." << std::endl;

  // Get solution and the Jacobian
  track->getCurrentState(sol);
  track->getStabilityMatrix(solJac);

  // Find eigenvalues
  gsl_vector_complex *eigVal = gsl_vector_complex_alloc(dim);
  gsl_eigen_nonsymm_workspace *w = gsl_eigen_nonsymm_alloc(dim);
  gsl_eigen_nonsymm(solJac, eigVal, w);

  // Print fixed point
  std::cout << "Fixed point:" << std::endl;
  gsl_vector_fprintf(stdout, sol, "%lf");
  std::cout << "Eigenvalues:" << std::endl;
  gsl_vector_complex_fprintf(stdout, eigVal, "%lf");
  
  gsl_eigen_nonsymm_free(w);
  gsl_vector_complex_free(eigVal);
  delete track;
  delete Jacobian;
  delete field;
  gsl_matrix_free(solJac);
  gsl_vector_free(sol);
  gsl_vector_free(x0);
  freeConfig();

  return 0;
}
