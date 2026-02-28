#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>

static const int COUNT = 4;
static const int ITERATION = 1000000;

long do_work(long k) {
    long x = 15;
    static const int nn = 87;

    for (long i = 1; i < nn; ++i)
        x = x / i + k % i;

    return x;
}

int cilk_main() {

    cilk::reducer< cilk::op_add<long> > arr[COUNT];

    cilk_for(int j = 0; j < ITERATION; j++) {
        for (int i = 0; i < COUNT; i++) {
            arr[i] += do_work(i + j);
        }
    }

    return 0;
}