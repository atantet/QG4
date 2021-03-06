#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cstring>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <math.h>
#include <libconfig.h++>
#include <ODESolvers.hpp>
#include <ODEFields.hpp>
#include <ODECont.hpp>
#include "../cfg/readConfig.hpp"

using namespace libconfig;


/** \file POContQG4.cpp 
 *  \ingroup examples
 *  \brief Fixed point continuation in quasi-geostrophic four modes model.
 *
 *  Fixed point continuation in quasi-geostrophic four modes model
 *  (Simonnet, Ghil, Dijkstra, 2005).
 */


/* *  \brief Fixed point continuation in quasi-geostrophic four modes model.
 *
 *  Fixed point continuation in quasi-geostrophic four modes model
 *  (Simonnet, Ghil, Dijkstra, 2005).
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

  gsl_matrix *solJac = gsl_matrix_alloc(dim + 1, dim + 1);
  gsl_vector_complex *eigVal = gsl_vector_complex_alloc(dim);
  gsl_matrix_complex *eigVec = gsl_matrix_complex_alloc(dim, dim);
  gsl_eigen_nonsymmv_workspace *w = gsl_eigen_nonsymmv_alloc(dim);
  gsl_matrix_view jac;
  char dstFileName[256], dstFileNameVec[256], dstFileNameVal[256], dstPostfix[256];
  FILE *dstStream, *dstStreamVec, *dstStreamVal;


  // Define names and open destination file
  double sigmaAbs = sqrt(sigmaStep*sigmaStep);
  double sign = sigmaStep / sigmaAbs;
  double exp = gsl_sf_log(sigmaAbs)/gsl_sf_log(10);
  double mantis = sign * gsl_sf_exp(gsl_sf_log(sigmaAbs) / exp);
  sprintf(dstPostfix, "%s_sigma%04d_sigmaStep%de%d", srcPostfix,
	  (int) (gsl_vector_get(initCont, dim) * 1000 + 0.1),
	  (int) (mantis*1.01), (int) (exp*1.01));
  sprintf(dstFileName, "%s/continuation/fpCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStream = fopen(dstFileName, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  sprintf(dstFileNameVec, "%s/continuation/fpEigVecCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamVec = fopen(dstFileNameVec, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  
  sprintf(dstFileNameVal, "%s/continuation/fpEigValCont%s.%s",
	  resDir, dstPostfix, fileFormat);
  if (!(dstStreamVal = fopen(dstFileNameVal, "w")))
    {
      fprintf(stderr, "Can't open %s for writing simulation: ", dstFileName);
      perror("");
      return EXIT_FAILURE;
    }
  
  // Define field
  std::cout << "Defining deterministic vector field..." << std::endl;
  vectorField *field = new QG4Cont(ci, li);
  
  // Define linearized field
  std::cout << "Defining Jacobian, initialized at initCont..." << std::endl;
  linearField *Jacobian = new JacobianQG4Cont(ci, li, initCont);

  // Define fixed point problem
  fixedPointCont *track = new fixedPointCont(field, Jacobian, eps, eps, maxIter);

  // First correct
  std::cout << "Applying initial correction..." << std::endl;
  track->correct(initCont);

  if (!track->hasConverged())
    {
      std::cerr << "First correction could not converge." << std::endl;
      return -1;
    }
  else
    std::cout << "Found initial fixed point after "
	      << track->getNumIter() << " iterations"
	      << " with distance = " << track->getDist()
	      << " and step = " << track->getStepCorrSize() << std::endl;


  while ((gsl_vector_get(initCont, dim) >= sigmaMin)
	 && (gsl_vector_get(initCont, dim) <= sigmaMax))
    {
      // Find fixed point
      std::cout << "\nApplying continuation step..." << std::endl;
      track->continueStep(sigmaStep);

      if (!track->hasConverged())
	{
	  std::cerr << "Continuation could not converge." << std::endl;
	  break;
	}
      else
	std::cout << "Found initial fixed point after "
		  << track->getNumIter() << " iterations"
		  << " with distance = " << track->getDist()
		  << " and step = " << track->getStepCorrSize() << std::endl;

      // Get solution and the Jacobian
      track->getCurrentState(initCont);
      track->getStabilityMatrix(solJac);
      jac = gsl_matrix_submatrix(solJac, 0, 0, dim, dim);

      // Find eigenvalues
      gsl_eigen_nonsymmv(&jac.matrix, eigVal, eigVec, w);

      // Print fixed point
      std::cout << "Fixed point:" << std::endl;
      gsl_vector_fprintf(stdout, initCont, "%lf");
      std::cout << "Valenvalues:" << std::endl;
      gsl_vector_complex_fprintf(stdout, eigVal, "%lf");

      // Write results
      if (strcmp(fileFormat, "bin") == 0)
	{
	  gsl_vector_fwrite(dstStream, initCont);
	  gsl_vector_complex_fwrite(dstStreamVal, eigVal);
	  gsl_matrix_complex_fwrite(dstStreamVec, eigVec);
	}
      else
	{
	  gsl_vector_fprintf(dstStream, initCont, "%lf");
	  gsl_vector_complex_fprintf(dstStreamVal, eigVal, "%lf");
	  gsl_matrix_complex_fprintf(dstStreamVec, eigVec, "%lf");
	}
      // Flush in case premature exit
      fflush(dstStream);
      fflush(dstStreamVal);
      fflush(dstStreamVec);
    }
  
  gsl_eigen_nonsymmv_free(w);
  gsl_vector_complex_free(eigVal);
  gsl_matrix_complex_free(eigVec);
  delete track;
  delete Jacobian;
  delete field;
  gsl_matrix_free(solJac);
  gsl_vector_free(initCont);
  fclose(dstStreamVal);
  fclose(dstStreamVec);
  fclose(dstStream);  
  freeConfig();

  return 0;
}
