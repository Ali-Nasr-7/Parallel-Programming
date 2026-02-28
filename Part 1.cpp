#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

const int GRAIN_SIZE = 4096;

long long parallel_sum(vector<int>& arr, int left, int right) {

    int size = right - left;

    if (size <= GRAIN_SIZE) {
        long long sum = 0;
        for (int i = left; i < right; i++)
            sum += arr[i];
        return sum;
    }

    int mid = left + size / 2;

    long long left_sum = 0;
    long long right_sum = 0;

#pragma omp parallel sections
    {
#pragma omp section
        {
            left_sum = parallel_sum(arr, left, mid);
        }

#pragma omp section
        {
            right_sum = parallel_sum(arr, mid, right);
        }
    }

    return left_sum + right_sum;
}

int main() {

    int n = 1 << 24;   
    vector<int> arr(n, 1);

    int thread_counts[] = { 1, 2, 4, 8, 16 };

    for (int t : thread_counts) {

        omp_set_num_threads(t);

        double start = omp_get_wtime();

        long long result = parallel_sum(arr, 0, n);

        double end = omp_get_wtime();

        double time_taken = end - start;

        cout << "Threads: " << t
            << " | Time: " << time_taken
            << " seconds"
            << endl;
    }

    return 0;
}