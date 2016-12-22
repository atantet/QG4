#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl_extension.hpp>
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

  gsl_vector *sol = gsl_vector_alloc(dim + 1);
  gsl_matrix *solFM = gsl_matrix_alloc(dim, dim);

  // Define field
  //std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new QG4(sigma, ci, li);
  
  // Define linearized field
  //std::cout << "Defining Jacobian, initialized at initCont..." << std::endl;
  linearField *Jacobian = new JacobianQG4(sigma, ci, li);

  // Define numerical scheme
  //std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim);
  //numericalScheme *scheme = new Euler(dim);

  // Define model (the initial state will be assigned later)
  //std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme);

  // Define linearized model 
  //std::cout << "Defining linearized model..." << std::endl;
  fundamentalMatrixModel *linMod = new fundamentalMatrixModel(mod, Jacobian);

  // Define fixed point problem
  std::cout << "Tracking periodic orbit..." << std::endl;
  periodicOrbitCorr *track = new periodicOrbitCorr(linMod, eps, eps, maxIter, dt,
						   numShoot);
  // Find fixed point
  track->findSolution(initCont);		       

  if (!track->hasConverged())
    std::cerr << "Did not converge." << std::endl;
  else
    std::cout << "Found periodic point after "
	      << track->getNumIter() << " iterations"
	      << " with distance = " << track->getDist()
	      << " and step = " << track->getStepCorrSize() << std::endl;

  // Get solution and the Jacobian
  track->getCurrentState(sol);
  track->getStabilityMatrix(solFM);

  // Find eigenvalues
  gsl_vector_complex *eigVal = gsl_vector_complex_alloc(dim);
  gsl_vector_complex *FloquetExp = gsl_vector_complex_alloc(dim);
  gsl_eigen_nonsymm_workspace *w = gsl_eigen_nonsymm_alloc(dim);
  gsl_eigen_nonsymm(solFM, eigVal, w);
  // Convert to Floquet exponents
  gsl_vector_complex_log(FloquetExp, eigVal);
  gsl_vector_complex_scale_real(FloquetExp, 1. / gsl_vector_get(sol, dim));

  // Print fixed point
  std::cout << "Periodic point and period:" << std::endl;
  gsl_vector_fprintf(stdout, sol, "%lf");
  std::cout << "Eigenvalues:" << std::endl;
  gsl_vector_complex_fprintf(stdout, FloquetExp, "%lf");
  
  gsl_eigen_nonsymm_free(w);
  gsl_vector_complex_free(FloquetExp);
  gsl_vector_complex_free(eigVal);
  delete track;
  delete linMod;
  delete mod;
  delete scheme;
  delete Jacobian;
  delete field;
  gsl_matrix_free(solFM);
  gsl_vector_free(sol);
  freeConfig();

  return 0;
}
