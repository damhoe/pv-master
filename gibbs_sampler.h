/** Gibbs sampler header file.
 *
 *
 * @author: Damian Hoedtke
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

double * CreateLocations(int L, int N);
//double PanelDensity(double x, double y, double r, double *locations, int *states, int N);
double PanelDensity(int *indices, int *states, int N);
int * populate(double *locs, double *economics,  int N);
int * populate_advertisement(double *locs, double *economics, int L, int N);
double * create_rnd(double min, double max, int N);
double ** distance_matrix(double * points, int size);
int ** GetIndices(double **distances, double r, int size);
