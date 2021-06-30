/** Gibbs sampler
 *
 * @author: Damian Hoedtke
 *
 */

#include "gibbs_sampler.h"

double * CreateLocations(int L, int N)
{
    // generate random locations within A = l * l

    // alloc memory for locations
    double *locations = malloc(sizeof(double) * 2*N);

    // iterate over locations
    for (int k=0; k<N; k++)
    {
        double x = (double) rand() / RAND_MAX * L;
        double y = (double) rand() / RAND_MAX * L;
        locations[2*k] = x;
        locations[2*k+1] = y;
    }

    return locations;
}

double PanelDensityOld(double x, double y, double r, double *locations, int *states, int N)
{
    // counter variables
    int n_locs = 0;
    int n_modules = 0;

    for (int k=0; k<N; k++)
    {
        double dx = x - locations[2*k];
        double dy = y - locations[2*k+1];

        if (dx * dx + dy * dy <= r)
        {
            n_locs += 1;
            n_modules += states[k];
        }
    }

    if (n_locs > 0) // avoid division by zero
    {
        return n_modules / n_locs;
    }

    return 0;
}

double PanelDensity(int *indices, int *state, int size)
{
    if (size < 1) // avoid division by zero
    {
        return 0;
    }

    int n_modules = 0;
    for (int i=0; i<size; i++)
    {
        if (indices[i] == -1)
        {
            break;
        }
        n_modules += state[indices[i]];
    }

    return n_modules / size;
}

/* compute pairwise distance between points */
double ** distance_matrix(double * points, int size)
{
    double **matrix = malloc(size * sizeof(double*));

    for (int k=0; k<size; k++)
    {
        matrix[k] = calloc(size, sizeof(double));
    }

    // calculate distances
    for (int i=0; i<size; i++)
    {
        for (int j=i+1; j<size; j++)
        {
            double x1 = points[2*i];
            double y1 = points[2*i+1];
            double x2 = points[2*j];
            double y2 = points[2*j+1];
            double dx = x2 - x1;
            double dy = y2 - y1;
            double d2 = dx * dx + dy * dy;
            matrix[i][j] = d2;
            matrix[j][i] = d2;
        }
    }

    return matrix;
}

/* Estimate the indices of all locations within radius r for each location. */
int ** GetIndices(double ** distances, double r, int size)
{
    double r2 = r * r;

    int **indices = malloc(size * sizeof(int*));
    int * helper = malloc(size * sizeof(int));

    for (int k=0; k<size; k++)
    {
        int count = 0;
        for (int j=0; j<size; j++)
        {
            if (distances[k][j] < r2)
            {
                helper[count] = j; // save index
                count += 1;
            }
        }

        count += 1; // for END flag

        // copy indices to array
        indices[k] = malloc(count * sizeof(int));
        indices[k][count-1] = -1; // END flag
        for (int i=0; i<count-1; i++)
        {
            indices[k][i] = helper[i];
        }
    }

    free(helper);

    return indices;
}

int * populate(double *locs, double *economics,  int N)
{
    // alloc memory
    int *state = calloc(N, sizeof(int));

    // initialize state
    double *helper = create_rnd(0, 1, N);
    double n0 = 0.2; // initial solar cell density
    int nPanels = 0;
    for (int j=0; j<N; j++)
    {
        state[j] = helper[j] < n0;
        nPanels += state[j];
    }

    printf("@---'populate':\n \t nGoal = %.2f, nReal=%.2f\n", n0, (double) nPanels/N);

    int nr = 5;
    double radii[5] = {0.5, 1, 1.5, 2, 3};
    double weight = 2.5;

    // for each radius, save the locations of relevant
    double **distances = distance_matrix(locs, N);

    int ***indices = malloc(nr * sizeof(int**));
    for (int i=0; i<nr; i++)
    {
        indices[i] = GetIndices(distances, radii[i], N);
    }

    int n_steps = 20;
    for (int i=0; i<n_steps; i++)
    {
        // ---------------
        // proceed MC step
        // ---------------
        for (int k=0; k<N; k++) // iterate over locations
        {
            // calculate pk
            double pk = 0.0;

            for (int l=0; l<nr; l++)
            {
                double r = radii[l];
                double beta = weight * exp(-r);
                double rho = PanelDensity(indices[l][k], state, N);
                pk += beta * (rho - economics[k]);
            }

            pk /= nr;
            pk += economics[k];

            // account for total density
            //pk += 10;//  * (n0 - (double) nPanels / N);

            // assign solar panel
            nPanels -= state[k];
            double rnd = (double) rand() / RAND_MAX;
            state[k] = pk > rnd;
            nPanels += state[k];
        }

        // print panel density
        printf("@-'populate':\t d_rho = %.6f\n", (double) nPanels / N - n0);
    }

    return state;
}

int * populate_advertisement(double *locs, double *economics, int L, int N)
{
    // alloc memory
    int *state = calloc(N, sizeof(int));

    double *ecoCpy = malloc(N * sizeof(double));
    memcpy(ecoCpy, economics, N * sizeof(double));

    // define some populate_advertisement areas
    // inside theses areas the prob. of having a solar cell doubles
    int n_areas = 50;
    double r = 1.0; // corresponds to x km
    double *areas = malloc(2*n_areas * sizeof(double));
    for (int m=0; m<n_areas; m++)
    {
        areas[2*m] = (double) rand() / RAND_MAX * L;
        areas[2*m+1] = (double) rand() / RAND_MAX * L;
    }

    double *advertisment = create_rnd(0.0, 1, n_areas);

    for (int k=0; k<N; k++)
    {
        for (int m=0; m<n_areas; m++)
        {
            double dx = areas[2*m] - locs[2*k];
            double dy = areas[2*m+1] - locs[2*k+1];

            if (dx * dx + dy * dy < r * r)
            {
                ecoCpy[k] += advertisment[m];
                break;
            }

            else if (dx * dx + dy * dy < 4 * r * r)
            {
                ecoCpy[k] += 0.5 * advertisment[m];
                break;
            }
        }

        double alpha = 1.0;

        if ( alpha * ecoCpy[k] + (1 - alpha) * 0.5 > 0.5)
        {
            state[k] = 1;
        }
    }
    free(areas);
    free(advertisment);

    return state;
}

double * create_rnd(double min, double max, int N)
{
    double *arr = malloc(N * sizeof(double));

    for (int i=0; i<N; i++)
    {
        arr[i] = (double) rand() / RAND_MAX * (max - min) + min;
    }
    return arr;
}
