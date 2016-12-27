#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_log.h>
#include <ergoGrid.hpp>
#include <transferOperator.hpp>
#include <gsl_extension.hpp>
#include "../cfg/readConfig.hpp"


/** \file transfer.cpp
 *  \ingroup examples
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
     Config cfg;
     std::cout << "Sparsing config file " << configFileName << std::endl;
     cfg.readFile(configFileName);
     readGeneral(&cfg);
     readModel(&cfg);
     readSimulation(&cfg);
     readSprinkle(&cfg);
     readGrid(&cfg);
     readTransfer(&cfg);
     std::cout << "Sparsing success.\n" << std::endl;
    }
  catch (...)
    {
      std::cerr << "Error reading configuration file" << std::endl;
      return(EXIT_FAILURE);
    }

  // Observable declarations
  char srcPostfix[256], srcFileName[256], simPostfix[256], dstGridPostfix[256];
  FILE *srcStream;
  gsl_matrix *initStates, *finalStates;
  gsl_vector_view stateView;

  // Grid declarations
  Grid *grid;
  char gridFileName[256];

  // Grid membership declarations
  char gridMemFileName[256];
  FILE *gridMemStream;
  gsl_matrix_uint *gridMemMatrix;
    
  // Transfer operator declarations
  char forwardTransitionFileName[256], initDistFileName[256],
    backwardTransitionFileName[256], finalDistFileName[256],
    postfix[256], maskFileName[256];

  transferOperator *transferOp;

  
  // Get grid membership matrix
  sprintf(srcPostfix, "_%s", caseName);
  if (! readGridMem)
    {
      // Define names and open source file
      sprintf(simPostfix, "%s%s_sigma%04d_L%d_spinup%d_dt%d_samp%d_nTraj%d",
	      srcPostfix, boxPostfix, (int) (sigma * 1000 + 0.1), (int) (L * 1000),
	      (int) spinup, (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1),
	      (int) printStepNum, nTraj);
      sprintf(srcFileName, "%s/simulation/sim%s.%s",
	      resDir, simPostfix, fileFormat);
      if (!(srcStream = fopen(srcFileName, "r")))
	{
	  fprintf(stderr, "Can't open %s for writing simulation: ",
		  srcFileName);
	  perror("");
	  return EXIT_FAILURE;
	}

      // Read trajectories
      std::cout << "Reading trajectory in " << srcFileName << std::endl;
      initStates = gsl_matrix_alloc(nTraj, dim);
      finalStates = gsl_matrix_alloc(nTraj, dim);
      for (size_t traj = 0; traj < (size_t) nTraj; traj++)
	{
	  if (strcmp(fileFormat, "bin") == 0)
	    {
	      stateView = gsl_matrix_row(initStates, traj);
	      gsl_vector_fread(srcStream, &stateView.vector);
	      stateView = gsl_matrix_row(finalStates, traj);
	      gsl_vector_fread(srcStream, &stateView.vector);
	    }
	  else
	    {
	      stateView = gsl_matrix_row(initStates, traj);
	      gsl_vector_fscanf(srcStream, &stateView.vector);
	      stateView = gsl_matrix_row(finalStates, traj);
	      gsl_vector_fscanf(srcStream, &stateView.vector);
	    }
	}
      
      // Close trajectory file
      fclose(srcStream);

      // Define grid
      grid = new RegularGrid(nx, gridLimitsLow, gridLimitsUp);
    
      // Print grid
      sprintf(gridFileName, "%s/grid/grid%s%s%s.txt", resDir, srcPostfix,
	      gridPostfix);
      grid->printGrid(gridFileName, "%.12lf", true);


      // Grid membership file name
      sprintf(dstGridPostfix, "%s%s_sigma%04d_L%d_spinup%d_dt%d_samp%d_nTraj%d",
	      srcPostfix, gridPostfix, (int) (sigma * 1000 + 0.1), (int) (L * 1000),
	      (int) spinup, (int) round(-gsl_sf_log(dt)/gsl_sf_log(10)+0.1),
	      (int) printStepNum, nTraj);
      sprintf(gridMemFileName, "%s/transfer/gridMem/gridMemMatrix%s.%s",
	      resDir, dstGridPostfix, fileFormat);
  
      // Open grid membership vector stream
      if ((gridMemStream = fopen(gridMemFileName, "w")) == NULL)
	{
	  fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	  perror("");
	  return(EXIT_FAILURE);
	}
    
      // Get grid membership vector
      std::cout << "Getting grid membership matrix..." << std::endl;
      gridMemMatrix = getGridMemMatrix(initStates, finalStates, grid);

      // Write grid membership matrix
      if (strcmp(fileFormat, "bin") == 0)
	gsl_matrix_uint_fwrite(gridMemStream, gridMemMatrix);
      else
	gsl_matrix_uint_fprintf(gridMemStream, gridMemMatrix, "%d");

      // Free states and close stream
      gsl_matrix_free(initStates);
      gsl_matrix_free(finalStates);
      fclose(gridMemStream);
      delete grid;
    }
  else
    {
      // Read grid membership matrix
      // Grid membership file name
      sprintf(gridMemFileName, "%s/transfer/gridMem/gridMemMatrix%s.%s",
	      resDir, dstGridPostfix, fileFormat);
	  
      // Open grid membership stream for reading
      std::cout << "Reading grid membership matrix..."
		<< " at " << gridMemFileName << std::endl;
	  
      if ((gridMemStream = fopen(gridMemFileName, "r")) == NULL)
	{
	  fprintf(stderr, "Can't open %s for writing:", gridMemFileName);
	  perror("");
	  return(EXIT_FAILURE);
	}
      
      // Read grid membership
      gridMemMatrix = gsl_matrix_uint_alloc(nTraj, 2);
      if (strcmp(fileFormat, "bin") == 0)
	gsl_matrix_uint_fread(gridMemStream, gridMemMatrix);
      else
	gsl_matrix_uint_fscanf(gridMemStream, gridMemMatrix);

      // Close stream
      fclose(gridMemStream);
    }
  

  // Get transition matrices for one lag
  sprintf(postfix, "%s_tau%03d", dstGridPostfix, (int) (L * 1000 + 0.1));

  std::cout << "\nConstructing transfer operator for a lag of "
	    << L << std::endl;


  // Get transition matrices as CSR
  std::cout << "Building stationary transfer operator..." << std::endl;
  transferOp = new transferOperator(gridMemMatrix, N, false);


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
      
  // Write backward transition matrix
  std::cout << "Writing backward transition matrix \
and final distribution..." << std::endl;
  sprintf(backwardTransitionFileName,
	  "%s/transfer/backwardTransition/backwardTransition%s.coo%s",
	  resDir, postfix, fileFormat);
  transferOp->printBackwardTransition(backwardTransitionFileName,
				      fileFormat, "%.12lf");

  // Write final distribution 
  sprintf(finalDistFileName,
	  "%s/transfer/finalDist/finalDist%s.%s",
	  resDir, postfix, fileFormat);
  transferOp->printFinalDist(finalDistFileName,
				 fileFormat, "%.12lf");
	
  // Free
  delete transferOp;
  gsl_matrix_uint_free(gridMemMatrix);
  freeConfig();
  
  return 0;
}

