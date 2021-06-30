/** Gibbs sampler.
 *
 * @author: Damian Hoedtke
 */
#include <stdio.h>
#include <stdlib.h>

#include "gibbs_sampler.h"

int main(int argc, char const *argv[]) {

    // set rnd seed
    srand(0);

    // define params
    int N = 10000;
    int L = 20; // 4 km

    // populate
    double *locs = CreateLocations(L, N);
    double *economics = create_rnd(0.5, 0.6, N);
    int *states = populate(locs, economics, N);
    //int *states = populate_advertisement(locs, economics, L, N);

    // print to a file
    FILE *fp;

    //fp = fopen("distribution_advertisement.txt", "w+");
    fp = fopen("distribution.txt", "w+");
    for (int k=0; k<N; k++)
    {
        fprintf(fp, "%.3f,%3f,%d,%.3f\n", locs[2*k], locs[2*k+1], states[k], economics[k]);
    }
    fclose(fp);

    free(locs);
    free(states);
    free(economics);

    return 0;
}
