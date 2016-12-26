#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include <ergoGrid.hpp>
#include <transferOperator.hpp>
#include <gsl_extension.hpp>
#include "../cfg/readConfig.hpp"


/** \file transfer.cpp
 *  \brief Get transition matrices and distributions directly from time series.
 *   
 * Get transition matrices and distributions from a long time series
 * (e.g. simulation output).
 * Takes as first a configuration file to be parsed with libconfig C++ library.
 * First read the observable and get its mean and standard deviation
 * used to adapt the grid.
 * A rectangular grid is used here.
 * A grid membership vector is calculated for each time series 
 * assigning to each realization a grid box.
 * Then, the membership matrix is calculated for a given lag.
 * The forward transition matrices as well as the initial distributions
 * are calculated from the membership matrix.
 * Note that, since the transitions are calculated from long time series,
 * the problem must be autonomous and ergodic (stationary) so that
 * the backward transition matrix and final distribution need not be calculated.
 * Finally, the results are printed.
 */


/** \brief Calculate transfer operators from time series.
 *
 *  After parsing the configuration file,
 *  the time series is read and an observable is designed
 *  selecting components with a given embedding lag.
 *  A membership vector is then built from the observable,
 *  attaching the box they belong to to every realization.
 *  The membership vector is then converted to a membership
 *  matrix for different lags and the transfer operators 
 *  built. The results are then written to file.
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

  // Observable declarations
  char dstGridPostfix[256];

  // Grid declarations
  Grid *grid;
  const int nTrajPerBox = nTraj / N;

  // Grid membership declarations
  gsl_matrix_uint *gridMem;
    
  // Transfer operator declarations
  char forwardTransitionFileName[256], initDistFileName[256],
    backwardTransitionFileName[256], finalDistFileName[256],
    postfix[256], maskFileName[256];

  transferOperator *transferOp;

  // Set random number generator
  gsl_rng * r = gsl_rng_alloc(gsl_rng_ranlxs1);
  // Get seed and set random number generator
  size_t seed = gsl_vector_uint_get(seedRng, 0);
  printf("Setting random number generator with seed: %d\n", (int) seed);
  gsl_rng_set(r, seed);

  
  ///
  /// Get grid membership matrix
  ///

  // Define grid and allocate grid membership matrix
  grid = new RegularGrid(nx, gridLimitsLow, gridLimitsUp);
  gridMem = gsl_matrix_uint_alloc(nTraj, 2);
  // Print grid
  grid->printGrid(gridFileName, "%.12lf", true);


  // Iterate for each trajectory
  std::cout << "Getting grid membership matrix..." << std::endl;
#pragma omp parallel
  {
    size_t boxf;
    
    // Initial condition and final box membership per initial box
    gsl_vector *IC = gsl_vector_alloc(dim);
    gsl_vector_uint *boxMem = gsl_vector_uint_alloc(nTrajPerBox);
    gsl_vector_uint *multiIdx = gsl_vector_uint_alloc(dim);
    gsl_vector *minBox = gsl_vector_alloc(dim);
    gsl_vector *maxBox = gsl_vector_alloc(dim);
    
    // Define field
    vectorField *field = new QG4(sigma, ci, li);
  
    // Define numerical scheme
    numericalScheme *scheme = new RungeKutta4(dim);

    // Define model (the initial state will be assigned later)
    model *mod = new model(field, scheme);

    // Srinkle box by box
#pragma omp for
    for (size_t box0 = 0; box0 < N; box0++)
      {
	// Verbose
	if (box0 % (N / 100) == 0)
	  {
#pragma omp critical
	    {
	      std::cout << "Getting transitions from box " << box0 << " of " << N-1
			<< std::endl;
	    }
	  }
	
	// Get bounds of box
	unravel_index(box0, nx, multiIdx);
	for (size_t d = 0; d < (size_t) dim; d++)
	  {
	    // Get lower limit of box0 for dim d from grid bounds
	    gsl_vector_set(minBox, d,
			   gsl_vector_get(grid->bounds->at(d),
					  gsl_vector_uint_get(multiIdx, d)));
	    // Get upper limit of box0 for dim d from grid bounds
	    gsl_vector_set(maxBox, d,
			   gsl_vector_get(grid->bounds->at(d),
					  gsl_vector_uint_get(multiIdx, d) + 1));
	  }

	// Simulate trajecories from uniformly sampled initial conditions in box
	for (size_t traj = 0; traj < (size_t) nTrajPerBox; traj++)
	  {
	    // Get random initial distribution
	    for (size_t d = 0; d < (size_t) dim; d++)
	      gsl_vector_set(IC, d, gsl_ran_flat(r, gsl_vector_get(minBox, d),
						 gsl_vector_get(maxBox, d)));

	    // Numerical integration
	    mod->integrateForward(IC, L, dt);

	    // Get box of final state
	    boxf = grid->getBoxMembership(mod->current);

	    // Add transition
	    gsl_vector_uint_set(boxMem, traj, boxf);	    
	  }

	// Save all transitions of box in grid membership matrix
#pragma omp critical
	{
	  // Copy final box from box0 to grid membership matrix
	  gsl_vector_uint_view gridMemCol \
	    = gsl_matrix_uint_subcolumn(gridMem, 1, box0*nTrajPerBox, nTrajPerBox);
	  gsl_vector_uint_memcpy(&gridMemCol.vector, boxMem);
	  // Copy box0 to grid membership matrix
	  gridMemCol \
	    = gsl_matrix_uint_subcolumn(gridMem, 0, box0*nTrajPerBox, nTrajPerBox);
	  gsl_vector_uint_set_all(boxMem, box0);
	  gsl_vector_uint_memcpy(&gridMemCol.vector, boxMem);
	}
      }
    gsl_vector_free(IC);
    gsl_vector_uint_free(boxMem);
    gsl_vector_free(minBox);
    gsl_vector_free(maxBox);
    delete mod;
    delete scheme;
    delete field;
  }
  // Free
  delete grid;

  // Get transition matrices for one lag
  // Grid membership postfix
  sprintf(dstGridPostfix, "%s_sigma%04d_L%d_dt%d_nTraj%d",
	  gridPostfix, (int) (sigma * 1000 + 0.1), (int) (L * 1000),
	  (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1), nTraj);
  sprintf(postfix, "%s_tau%03d", dstGridPostfix, (int) (L * 1000 + 0.1));

  std::cout << "\nConstructing transfer operator for a lag of "
	    << L << std::endl;


  // Get transition matrices as CSR
  std::cout << "Building stationary transfer operator..." << std::endl;
  transferOp = new transferOperator(gridMem, N, true);


  // Write results
  // Write forward transition matrix
  std::cout << "Writing forward transition matrix..."
	    << std::endl;
  sprintf(forwardTransitionFileName,
	  "%s/transfer/forwardTransition/forwardTransition%s.coo%s",
	  resDir, postfix, fileFormat);
  transferOp->printForwardTransition(forwardTransitionFileName,
				     fileFormat, "%.12lf");

  // Write mask and initial distribution
  sprintf(maskFileName, "%s/transfer/mask/mask%s.%s",
	  resDir, postfix, fileFormat);
  transferOp->printMask(maskFileName,
			fileFormat, "%.12lf");
  
  sprintf(initDistFileName, "%s/transfer/initDist/initDist%s.%s",
	  resDir, postfix, fileFormat);
  transferOp->printInitDist(initDistFileName,
			    fileFormat, "%.12lf");
      
//   // Write backward transition matrix
//   std::cout << "Writing backward transition matrix \
// and final distribution..." << std::endl;
//   sprintf(backwardTransitionFileName,
// 	  "%s/transfer/backwardTransition/backwardTransition%s.coo%s",
// 	  resDir, postfix, fileFormat);
//   transferOp->printBackwardTransition(backwardTransitionFileName,
// 				      fileFormat, "%.12lf");

//   // Write final distribution 
//   sprintf(finalDistFileName,
// 	  "%s/transfer/finalDist/finalDist%s.%s",
// 	  resDir, postfix, fileFormat);
//   transferOp->printFinalDist(finalDistFileName,
// 				 fileFormat, "%.12lf");
	
  // Free
  delete transferOp;
  gsl_matrix_uint_free(gridMem);
  freeConfig();
  
  return 0;
}

