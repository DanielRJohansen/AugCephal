#include "Toolbox.h"
#include <algorithm>


int Toolbox::find_kth(float* v, int n, int k, int depth) {
    if (n == 1 && k == 0) return v[0];

    int m = (n + 4) / 5;
    float* medians = new float[m];
    for (int i = 0; i < m; i++) {
        if (5 * i + 4 < n) {
            float* w = v + 5 * i;
            for (int j0 = 0; j0 < 3; j0++) {
                int jmin = j0;
                for (int j = j0 + 1; j < 5; j++) {
                    if (w[j] < w[jmin]) jmin = j;
                }
                swap(w[j0], w[jmin]);
            }
            medians[i] = w[2];
        }
        else {
            medians[i] = v[5 * i];
        }
    }

    int pivot = find_kth(medians, m, m / 2);
    delete[] medians;

    for (int i = 0; i < n; i++) {
        if (v[i] == pivot) {
            swap(v[i], v[n - 1]);
            break;
        }
    }

    int store = 0;
    for (int i = 0; i < n - 1; i++) {
        if (v[i] < pivot) {
            swap(v[i], v[store++]);
        }
    }
    swap(v[store], v[n - 1]);

    if (store == k) {
        return pivot;
    }
    else if (store > k) {
        return find_kth(v, store, k, depth+1);
    }
    else {
        return find_kth(v + store + 1, n - store - 1, k - store - 1, depth+1);
    }
}