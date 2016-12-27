#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file simQG4Sprinkle.cpp 
 *  \brief Simulate many trajectories of quasi-geostrophic double gyre model.
 *
 *  Simulate many trajectories of quasi-geostrophic double gyre model
 *  truncated to 4 modes (Simmonnet, Ghil, Dijkstra, 2005).
 */


/* *  \brief Simulate many trajectories of quasi-geostrophic double gyre model.
 *
 *  Simulate many trajectories quasi-geostrophic double gyre model
 *  truncated to 4 modes (Simmonnet, Ghil, Dijkstra, 2005).
 *  After parsing the configuration file,
 *  the vector field of the QG4 flow and the Runge-Kutta numerical scheme of order 4 are defined.
 *  The model is then integrated forward and the results saved.
 */
int main(int argc, char * argv[])
{
  FILE *dstStream;
  char srcPostfix[256], dstFileName[256], dstPostfix[256];
  size_t seed;

  // Read configuration file
  std::cout.precision(6);
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
      Config cfg;
      std::cout << "Sparsing config file " << configFileName << std::endl;
      cfg.readFile(configFileName);
      readGeneral(&cfg);
      readModel(&cfg);
      readSimulation(&cfg);
      readSprinkle(&cfg);
      std::cout << "Sparsing success.\n" << std::endl;
    }
  catch (...)
    {
      std::cerr << "Error reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }

  // Set random number generator
  gsl_rng * r = gsl_rng_alloc(gsl_rng_ranlxs1);
  // Get seed and set random number generator
  seed = gsl_vector_uint_get(seedRng, 0);
  printf("Setting random number generator with seed: %d\n", (int) seed);
  gsl_rng_set(r, seed);

  // Define names and open destination file
  sprintf(srcPostfix, "_%s%s", caseName, delayName);
  sprintf(dstPostfix, "%s%s%s_sigma%04d_L%d_spinup%d_dt%d_samp%d_nTraj%d",
	  srcPostfix, obsName, boxPostfix,
	  (int) (sigma * 1000 + 0.1), (int) (L * 1000), (int) spinup,
	  (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1), (int) printStepNum,
	  nTraj);
  sprintf(dstFileName, "%s/simulation/sim%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStream = fopen(dstFileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }

  // Iterate for each trajectory
#pragma omp parallel
  {
    gsl_matrix *X;
    gsl_vector *IC = gsl_vector_alloc(dim);
    
    // Define field
    vectorField *field = new QG4(sigma, ci, li);
  
    // Define numerical scheme
    numericalScheme *scheme = new RungeKutta4(dim);

    // Define model (the initial state will be assigned later)
    model *mod = new model(field, scheme);

#pragma omp for
    for (size_t traj = 0; traj < (size_t) nTraj; traj++)
      {
	// Get random initial distribution
	for (size_t d = 0; d < (size_t) dim; d++)
	  gsl_vector_set(IC, d,
			 gsl_ran_flat(r, gsl_vector_get(minInitState, d),
				      gsl_vector_get(maxInitState, d)));

	// Numerical integration
	X = gsl_matrix_alloc(1, 1); // False allocation will be corrected
	mod->integrateForward(IC, L, dt, spinup, printStepNum, &X);

	// Write results (one trajectory after the other)
#pragma omp critical
	{
	  if (strcmp(fileFormat, "bin") == 0)
	    gsl_matrix_fwrite(dstStream, X);
	  else
	    gsl_matrix_fprintf(dstStream, X, "%f");
	  fflush(dstStream);
	}

	// Free
	gsl_matrix_free(X);
      }
    gsl_vector_free(IC);
    delete mod;
    delete scheme;
    delete field;
  }
  
  fclose(dstStream);  
  gsl_rng_free(r);
  freeConfig();

  return 0;
}
