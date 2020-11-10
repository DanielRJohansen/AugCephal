#include "VolumeMaker.h"





VolumeMaker::VolumeMaker() {
    volume = new Block[VOL_X * VOL_Y * VOL_Z];
    loadScans();
    CudaOperator CudaOps;
    CudaOps.medianFilter(copyVolume(volume), volume);

    categorizeBlocks();
    open(0);    // For ref: Category cats[6] = {lung, fat, fluids, muscle, clot, bone };
    printf("\n");
}

void read_directory(const string& name, stringvec& v)
{
    string pattern(name);
    pattern.append("\\*");
    WIN32_FIND_DATA data;
    HANDLE hFind;
    if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
        do {
            v.push_back(data.cFileName);
        } while (FindNextFile(hFind, &data) != 0);
        FindClose(hFind);
    }
}

void VolumeMaker::loadScans() {
    stringvec v;
    read_directory(folder_path, v);
    for (int i = 2; i < VOL_Z+2; i++) {
        string im_path = folder_path;
        im_path.append(v[i]);
        cout << '\r' << im_path << endl;

        Mat img = imread(im_path, cv::IMREAD_UNCHANGED);
        int z = i - 2;
        if (img.empty()) {
            cout << "imload failed" << endl;
            return;
        }
        insertImInVolume(img, z);
    }
}

void VolumeMaker::insertImInVolume(Mat img, int z) {
    
    float norm_key = 1. / (HU_MAX - HU_MIN);
    Mat img_ = cv::Mat::zeros(Size(512, 512), CV_8UC1);
    for (int y = 0; y < img.cols; y++) {
        for (int x = 0; x < img.rows; x++) {
            double hu = img.at<uint16_t>(y, x) - 32768;
            if (hu < HU_MIN) {
                volume[xyzToIndex(x, y, z)].air = true;
                volume[xyzToIndex(x, y, z)].value = 0;
            }
            else if (hu > HU_MAX) {
                volume[xyzToIndex(x, y, z)].value = 1;
                volume[xyzToIndex(x, y, z)].bone = true;

            }
            else if (hu > 100 && hu < 300) {
                volume[xyzToIndex(x, y, z)].soft_tissue = true;
                volume[xyzToIndex(x, y, z)].value = (hu - HU_MIN) * norm_key;

            }
            else if (hu > -120 && hu < -90) {
                volume[xyzToIndex(x, y, z)].fat = true;
                volume[xyzToIndex(x, y, z)].value = (hu - HU_MIN) * norm_key;
            }
            else 
                volume[xyzToIndex(x, y, z)].value = (hu - HU_MIN) * norm_key;
            
            //volume[xyzToIndex(x, y, z)].cluster->mean = volume[xyzToIndex(x, y, z)].value;
        }
    }
}

Block* VolumeMaker::copyVolume(Block* from) {
    Block* to = new Block[VOL_X * VOL_Y * VOL_Z];
    for (int i = 0; i < VOL_X * VOL_Y * VOL_Z; i++) {
        to[i] = from[i];
    }
    return to;
}

void VolumeMaker::cluster() {
    Block* vol_copy = copyVolume(volume);
    for (int z = 0; z < VOL_Z; z++) {
        printf("Clustering Z %d  \n", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                
                for (int zoff = 0; zoff < 2; zoff++) {
                    for (int yoff = 0; yoff < 2; yoff++) {
                        for (int xoff = 0; xoff < 2; xoff++) {
                            if (xoff + yoff + zoff == 0) // Cannot cluster with itself
                                continue;
                            float clusterdif = vol_copy[xyzToIndex(x + xoff, y + yoff, z + zoff)].cluster->mean -
                                vol_copy[xyzToIndex(x, y, z)].cluster->mean;
                            if (abs(clusterdif) > CLUSTER_MAX_SEP) {

                            }
                        }
                    }
                }
            }
        }
    }
}

void VolumeMaker::categorizeBlocks() {
    for (int z = 0; z < VOL_Z; z++) {
        printf(" \r Categorizing z level %d  \n", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                int hu_index =(int) (volume[block_index].value * (colorscheme.upper_limit- colorscheme.lower_limit -1));
                //if (z == 2)
                  //  cout << block_index << " " << hu_index  << "  " << volume[block_index].value << endl;               
                volume[block_index].color = colorscheme.colors[hu_index];
                volume[block_index].cat_index = colorscheme.cat_indexes[hu_index];
            }
        }
    }
}

void VolumeMaker::open(int cat_i) {
    dilate(cat_i);
    erode(cat_i);
    assignColorFromCat();
}
void VolumeMaker::close(int cat_i) {
    erode(cat_i);
    dilate(cat_i);
    assignColorFromCat();
}
void VolumeMaker::dilate(int cat_i) {
    printf("\n");
    for (int z = 0; z < VOL_Z; z++) {
        printf(" \r Dilating z level %d", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                if (volume[block_index].cat_index = cat_i) {
                    for (int z_ = -1; z_ < 2; z_++) {
                        for (int y_ = -1; y_ < 2; y_++) {
                            for (int x_ = -1; x_ < 2; x_++) {
                                int block_index_ = xyzToIndex(x+x_, y+y_, y+z_);
                                volume[block_index_].cat_index = cat_i;
                            }                               
                        }
                    }
                }
            }
        }
    }
}

void VolumeMaker::erode(int cat_i) {
    printf("\n");
    for (int z = 0; z < VOL_Z; z++) {
        printf(" \r Eroding z level %d", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);

                if (volume[block_index].cat_index = cat_i) {
                    for (int z_ = -1; z_ < 2; z_++) {
                        for (int y_ = -1; y_ < 2; y_++) {
                            for (int x_ = -1; x_ < 2; x_++) {
                                int block_index_ = xyzToIndex(x+x_, y+y_, z+z_);
                                if (volume[block_index_].cat_index != cat_i) {
                                    volume[block_index_].cat_index = volume[block_index_].prev_cat_index;
                                    goto EROSION_DONE;
                                }
                            }
                        }
                    }
                }
            EROSION_DONE:
                int a = 0;
            }
        }
    }
}
void VolumeMaker::assignColorFromCat() {
    for (int i = 0; i < VOL_Z * VOL_Y * VOL_X; i++) {
        volume[i].color = colorscheme.categories[volume[i].cat_index].color;
        cout << volume[i].cat_index << "     " << volume[i].color.r << endl;
    }
}


