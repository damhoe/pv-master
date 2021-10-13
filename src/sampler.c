/** Gibbs sampler.
 *
 * @author: Damian Hoedtke
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double * CreateRandomNumbers(double min, double max, int N);
int *CreateState(double n0, int N);
void UpdateState(double *locations, int *state, int N);


int main(int argc, char const *argv[]) {

    // set rnd seed
    srand(0);

    // define params
    int N = 10000;
    int L = 20; // 4 km
    double n0 = 0.01;

    // populate
    double *locations = CreateRandomNumbers(0, L, 2*N); // 2*N due to two coordinates
    int *state = CreateState(n0, N);

    clock_t begin = clock();
    for (int i=0; i<5; i++)
    {
        UpdateState(locations, state, N);
        fprintf(mFile, "%.3f,%.3f\n", result[2], result[1])
    }
    clock_t end = clock();
    printf("%.4f", (double) (end - begin)/CLOCKS_PER_SEC);

    // print to a file
    FILE *fp;;

    fp = fopen("data/distribution.csv", "w+");
    fprintf(fp, "locx,locy,state\n");
    for (int k=0; k<N; k++)
    {
        fprintf(fp, "%.3f,%.3f,%d\n", locations[2*k], locations[2*k+1], state[k]);
    }
    fclose(fp);
;
    free(locations);
    free(state);

    return 0;
}


void Eval2(double *dLimits, double *locations, int *state, int N)
{
    // for all cells:
    // calculate density in radius d
    // take average
    // clean
    int *cells = malloc(sizeof(int) * N);
    int nCells = 0;
    int *empty = malloc(sizeof(int) * N);
    int nEmpty = 0;
    for (int k=0; k<N; k++)
    {
        if (state[k] == 0) {
            empty[nEmpty] = k;
            nEmpty += 1;
        } else {
            state[k] = 1;
            cells[nCells] = k;
            nCells += 1;
        }
    }

    for (int k=0; k<nCells; k++)
    {
        for (int j=0; j<nEmpty; j++)
        {
            int i1 = 2*empty[k];
            int i2 = 2*cells[j];
            double x1 = locations[i1];
            double y1 = locations[i1 + 1];
            double x2 = locations[i2];
            double y2 = locations[i2 + 1];
            double d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);

            if (dmin < 0 || d < dmin) {
                dmin = d;
            }
        }

        double p = scale * exp(-dmin);
        double rnd = (double) rand() / RAND_MAX;
        if (p > rnd) {
            state[k] = 2;
        }
    }
}


void UpdateState(double *locations, int *state, int N)
{
    double scale = 0.05; // controls diffusion speed

    // for each empty location:
    // calculate min dist to cell and install with p = scale * exp(-d_min)

    // clean
    int *cells = malloc(sizeof(int) * N);
    int nCells = 0;
    int *empty = malloc(sizeof(int) * N);
    int nEmpty = 0;
    for (int k=0; k<N; k++)
    {
        if (state[k] == 0) {
            empty[nEmpty] = k;
            nEmpty += 1;
        } else {
            state[k] = 1;
            cells[nCells] = k;
            nCells += 1;
        }
    }

    double dmin = -1; // init smaller zero
    printf("%d\n", nCells);
    for (int k=0; k<nEmpty; k++)
    {
        for (int j=0; j<nCells; j++)
        {
            int i1 = 2*empty[k];
            int i2 = 2*cells[j];
            double x1 = locations[i1];
            double y1 = locations[i1 + 1];
            double x2 = locations[i2];
            double y2 = locations[i2 + 1];
            double d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);

            if (dmin < 0 || d < dmin) {
                dmin = d;
            }
        }

        double p = scale * exp(-dmin);
        double rnd = (double) rand() / RAND_MAX;
        if (p > rnd) {
            state[k] = 2;
        }
    }

    free(cells);
    free(empty);
}


int *CreateState(double n0, int N)
{
    int *state = malloc(sizeof(int) * N);
    double *helper = CreateRandomNumbers(0, 1, N);

    for (int k=0; k<N; k++)
    {
        if (helper[k] < n0) {
            state[k] = 1;
        } else {
            state[k] = 0;
        }
    }

    return state;
}


double * CreateRandomNumbers(double min, double max, int N)
{
    // generate random locations within A = L * L

    // alloc memory for locations
    double *locations = malloc(sizeof(double) * N);

    // iterate over locations
    for (int k=0; k<N; k++)
    {
        double x = (double) rand() / RAND_MAX * (max - min) + min;
        locations[k] = x;
    }

    return locations;
}
