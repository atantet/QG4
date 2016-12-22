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
  FILE *dstStream;
  gsl_matrix *X;
  char dstFileName[256], dstPostfix[256];
  size_t seed;

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

  // Set random number generator
  gsl_rng * r = gsl_rng_alloc(gsl_rng_ranlxs1);

  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new QG4(sigma, ci, li);
  
  // Define numerical scheme
  std::cout << "Defining deterministic numerical scheme..." << std::endl;
  numericalScheme *scheme = new RungeKutta4(dim);
  //numericalScheme *scheme = new Euler(dim);

  // Define model (the initial state will be assigned later)
  std::cout << "Defining deterministic model..." << std::endl;
  model *mod = new model(field, scheme);

  // Iterate one simulation per seed
  for (size_t s = 0; s < nSeeds; s++)
    {
      // Get seed and set random number generator
      seed = gsl_vector_uint_get(seedRng, s);
      printf("Setting random number generator with seed: %d\n", (int) seed);
      gsl_rng_set(r, seed);

      // Define names and open destination file
      sprintf(dstPostfix, "%s_sigma%04d_L%d_spinup%d_dt%d_samp%d",
	      srcPostfix, (int) (sigma * 1000 + 0.1), (int) L, (int) spinup,
	      (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1), (int) printStepNum);
      // sprintf(dstFileName, "%s/simulation/sim%s_seed%d.%s",
      // 	      resDir, srcPostfix, (int) seed, fileFormat);
      sprintf(dstFileName, "%s/simulation/sim%s.%s",
      	      resDir, dstPostfix, fileFormat);
      if (!(dstStream = fopen(dstFileName, "w")))
	{
	  fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
	  perror("");
	  return EXIT_FAILURE;
	}

      // Get random initial distribution
      // gsl_vector_set(initState, 0, gsl_ran_flat(r, -1, 1));
      // gsl_vector_set(initState, 1, gsl_ran_flat(r, -1, 1));
      // gsl_vector_set(initState, 2, gsl_ran_flat(r, -1, 1));
      // gsl_vector_set(initState, 3, gsl_ran_flat(r, -1, 1));

      // Set initial state
      printf("Setting initial state to (%.1lf, %.1lf, %.1lf, %.1lf)\n",
	     gsl_vector_get(initState, 0),
	     gsl_vector_get(initState, 1),
	     gsl_vector_get(initState, 2),
	     gsl_vector_get(initState, 3));

      // Numerical integration
      printf("Integrating simulation...\n");
      X = gsl_matrix_alloc(1, 1); // False allocation will be corrected
      mod->integrateForward(initState, L, dt, spinup, printStepNum, &X);

      // Write results
      printf("Writing...\n");
      if (strcmp(fileFormat, "bin") == 0)
	gsl_matrix_fwrite(dstStream, X);
      else
	gsl_matrix_fprintf(dstStream, X, "%f");
      fclose(dstStream);  

      // Free
      gsl_matrix_free(X);
    }
  delete mod;
  delete scheme;
  delete field;
  gsl_rng_free(r);
  freeConfig();

  return 0;
}
