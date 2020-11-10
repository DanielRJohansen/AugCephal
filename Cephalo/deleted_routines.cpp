/*


void VolumeMaker::medianFilter() {
    Block* vol_copy = copyVolume(volume);
    for (int z = 0; z < VOL_Z; z++) {
        printf("Filtering layer %d  \n", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {

                int block_index = xyzToIndex(x, y, z);
                if (x * y * z == 0 || x == VOL_X - 1 || y == VOL_Y - 1 || z == VOL_Z - 1) {    // Set all edges to air to no (out of mem problems)
                    volume[block_index].air = true;
                }
                else {
                    //float window_values[27];
                    vector <float>window(27);
                    int i = 0;
                    for (int z_off = -1; z_off < 2; z_off++) {
                        for (int y_off = -1; y_off < 2; y_off++) {
                            for (int x_off = -1; x_off < 2; x_off++) {
                                window[i] = vol_copy[xyzToIndex(x + x_off, y + y_off, z + z_off)].value;
                                //cout << window[i] << endl;
                                i++;
                            }
                        }
                    }
                    sort(window.begin(), window.end());
                    volume[block_index].value = window[14];
                }
            }
        }
    }
}








*/