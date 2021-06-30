/** Unit test of 'gibbs_sampler.c'
 *
 *
 * @author: Damian Hoedtke
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "gibbs_sampler.h"

int DistanceMatrix_Speed_PrintTime()
{
    clock_t start, end;
    int n = 4;
    int sizes[4] = {10, 100, 1000, 10000};

    printf("@:'test_distance_matrix_speed'\n");

    for (int i=0; i<n; i++)
    {
        int size = sizes[i];
        double *points = calloc(2*size, sizeof(double));

        start = clock();
        double **matrix = distance_matrix(points, size);
        end = clock();

        double time = (double) (end - start) / CLOCKS_PER_SEC;

        // print to console
        printf("--- n=%d,\t time elapsed: %.4f s\n", size, time);

        // free memory
        free(points);
        for (int j=0; j<size; j++)
        {
            free(matrix[j]);
        }
        free(matrix);
    }

    return 0;
}

int Populate_NumberOfPanelsConstant_PrintRelativeDeviation()
{


    return 0;
}

int main(int argc, char const *argv[]) {

    int test_count = 0;
    int err_count = 0;

    err_count += DistanceMatrix_Speed_PrintTime();
    test_count += 1;

    err_count += Populate_NumberOfPanelsConstant_PrintRelativeDeviation();
    test_count += 1;

    printf("---Gibbs sampler unit test.----\n" \
        "\t %d tests executed.\t %d tests failed.\n", test_count, err_count);

    return 0;
}
