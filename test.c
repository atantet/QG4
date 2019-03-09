#include <gsl/gsl_spmatrix.h>

int main(int argc, char *argv[])
{
  gsl_spmatrix *T =  gsl_spmatrix_alloc_nzmax(10*10*10*10, 10*10*10*10, (size_t) atoi(argv[1]),
					      GSL_SPMATRIX_TRIPLET);

  gsl_spmatrix_free(T);
  
  return 0;
}

