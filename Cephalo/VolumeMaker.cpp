#include "VolumeMaker.h"


#include <iostream>
#include <fstream>  




VolumeMaker::VolumeMaker() {

    volume = new Block[VOL_X * VOL_Y * VOL_Z];
    loadScans();
    CudaOperator CudaOps;
    CudaOps.medianFilter(copyVolume(volume), volume);
    categorizeBlocks();   
    open(5);
    //close(2);
    open(1);
    cluster(5, 20);
    cluster(1, 10);
    cluster(2, 10);
    //open(5);    // For ref: Category cats[6] = {lung, fat, fluids, muscle, clot, bone };
    //close(0);
    //open(1);
    //open(3);
    //open(4);
    vector<int> ignores = {0,2,4};

    setIgnores(ignores);
    assignColorFromCat();
    printf("\n");
}
std::ofstream outfile("E:\\NormImages\\gl3.txt");

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
void saveNormIm(Mat im, int number) {
    float norm_key = 1. / (700 +200);
    int a;
    for (int y = 0; y < im.cols; y++) {
        for (int x = 0; x < im.rows; x++) {
            double hu = im.at<uint16_t>(y, x) - 32768;
            if (hu < HU_MIN) {
                im.at<uint16_t>(y, x) = 0;
                a = 0;
            }
            else if (hu > HU_MAX) {
                im.at<uint16_t>(y, x) = 1 * 65500;
                a = 1 * 65500;
            }
            else {
                im.at<uint16_t>(y, x) = (hu - HU_MIN) * norm_key * 65500;
                a = (hu - HU_MIN) * norm_key * 65500;
            }
            outfile << to_string(a) + ' ';
        }
    }
    //imwrite("E:\\NormImages\\" + to_string(number) + ".png", im);
}
void VolumeMaker::loadScans() {
    stringvec v;
    read_directory(folder_path, v);
    for (int i = 2; i < VOL_Z+2; i++) {
        string im_path = folder_path;
        im_path.append(v[i]);
        cout << '\r' << im_path;

        Mat img = imread(im_path, cv::IMREAD_UNCHANGED);
        int z = i - 2;
        if (img.empty()) {
            cout << "imload failed" << endl;
            return;
        }
        //saveNormIm(img, i - 2);
        insertImInVolume(img, z);
    }
    cout << endl;
    outfile.close();

}



void VolumeMaker::insertImInVolume(Mat img, int z) { 
    float norm_key = 1. / (HU_MAX - HU_MIN);
    Mat img_ = cv::Mat::zeros(Size(512, 512), CV_8UC1);
    for (int y = 0; y < img.cols; y++) {
        for (int x = 0; x < img.rows; x++) {
            double hu = img.at<uint16_t>(y, x) - 32768;
            if (hu < HU_MIN) {
                volume[xyzToIndex(x, y, z)].ignore = true;
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

int VolumeMaker::xyzToIndex(int x, int y, int z) { return z * 512 * 512 + y * 512 + x; }
bool VolumeMaker::isLegal(int x, int y, int z) { return x >= 0 && y >= 0 && z >= 0 && x < VOL_X&& y < VOL_Y&& z < VOL_Z; }
bool VolumeMaker::isNotClustered(int block_index) { return volume[block_index].cluster_id == -1; }
bool VolumeMaker::isCategory(int block_index, int cat_id) { return volume[block_index].cat_index == cat_id; }

void VolumeMaker::cluster(int category_index, int min_cluster_size) {               //Currently not in use
    int cluster_id = 0;
    int clustersize;
    vector<Cluster> clusters;
    for (int z = 0; z < VOL_Z; z++) {
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                if (isCategory(block_index, category_index) && isNotClustered(block_index)) {
                    clustersize = propagateCluster(x, y, z, cluster_id, category_index, 1);    // DEEP recursive function
                    clusters.push_back(Cluster(cluster_id, clustersize));
                    cluster_id++;
                    printf(" \r Clusters of %s : %d ", colorscheme.category_ids[category_index].c_str(), cluster_id);
                }
            }
        }
    }
    for (int z = 0; z < VOL_Z; z++) {
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                if (volume[block_index].cluster_id != 0 && volume[block_index].cat_index == category_index) {
                    if (clusters[volume[block_index].cluster_id].size < min_cluster_size)
                        volume[block_index].ignore = true;
                }              
            }
        }
    }
}

int VolumeMaker::propagateCluster(int x, int y, int z, int cluster_id, int category_index, int depth) {
    int clustersize = 1;
    volume[xyzToIndex(x, y, z)].cluster_id = cluster_id;

    for (int z_off = -1; z_off < 2; z_off++) {
        for (int y_off = -1; y_off < 2; y_off++) {
            for (int x_off = -1; x_off < 2; x_off++) {
                int x_ = x + x_off;
                int y_ = y + y_off;
                int z_ = z + z_off;
                if (isLegal(x_, y_, z_)) {
                    int block_index = xyzToIndex(x_, y_, z_);
                    if (isCategory(block_index, category_index) && isNotClustered(block_index)) {
                        clustersize += propagateCluster(x_, y_, z_, cluster_id, category_index, depth+1);
                    }
                }
            }
        }
    }
    return clustersize;
}

void VolumeMaker::categorizeBlocks() {
    for (int z = 0; z < VOL_Z; z++) {
        printf(" \r Categorizing z level %d ", z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                int hu_index =(int) (volume[block_index].value * (colorscheme.upper_limit- colorscheme.lower_limit -1));
                volume[block_index].cat_index = colorscheme.cat_indexes[hu_index];
                volume[block_index].prev_cat_index = volume[block_index].cat_index;
            }
        }
    }
}

void VolumeMaker::open(int cat_i) {
    dilate(cat_i);
    erode(cat_i);
    //assignColorFromCat();
    updatePreviousCat();
}
void VolumeMaker::close(int cat_i) {
    erode(cat_i);
    dilate(cat_i);
    //assignColorFromCat();
    updatePreviousCat();
}
void VolumeMaker::dilate(int cat_i) {
    printf("\n");
    for (int z = 0; z < VOL_Z; z++) {
        printf(" \r Dilating %s %d", colorscheme.category_ids[cat_i].c_str(), z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                if (volume[block_index].prev_cat_index == cat_i) {
                    for (int z_ = -1; z_ < 2; z_++) {
                        for (int y_ = -1; y_ < 2; y_++) {
                            for (int x_ = -1; x_ < 2; x_++) {
                                int block_index_ = xyzToIndex(x + x_, y + y_, z + z_);
                                if (block_index_ >= 0 && block_index_ < VOL_X * VOL_Y * VOL_Z) {
                                    //volume[block_index_].prev_cat_index = volume[block_index_].cat_index;
                                    volume[block_index_].cat_index = cat_i;
                                }
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
        printf(" \r Eroding %s %d", colorscheme.category_ids[cat_i].c_str(), z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                if (volume[block_index].cat_index == cat_i) {
                    for (int z_ = -1; z_ < 2; z_++) {
                        for (int y_ = -1; y_ < 2; y_++) {
                            for (int x_ = -1; x_ < 2; x_++) {
                                int block_index_ = xyzToIndex(x+x_, y+y_, z+z_);
                                if (block_index_ > 0 && block_index_ < VOL_X * VOL_Y * VOL_Z) {
                                    if (volume[block_index_].cat_index != cat_i) {
                                        volume[block_index].cat_index = volume[block_index].prev_cat_index;
                                        goto EROSION_DONE;
                                    }
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
void VolumeMaker::updatePreviousCat() {
    for (int i = 0; i < VOL_Z * VOL_Y * VOL_X; i++) {
        volume[i].prev_cat_index = volume[i].cat_index;
    }
}

void VolumeMaker::setIgnores(vector<int> ignores) {
    for (int i = 0; i < VOL_Z * VOL_Y * VOL_X; i++) {
        for (int j = 0; j < ignores.size(); j++) {
            if (volume[i].cat_index == ignores[j]) {
                volume[i].ignore = true;
                break;
            }
        }
    }
}

void VolumeMaker::assignColorFromCat() {
    for (int i = 0; i < VOL_Z * VOL_Y * VOL_X; i++) {
        //cout << i << "  " <<volume[i].cat_index << endl;
        volume[i].color = colorscheme.categories[volume[i].cat_index].color;
        //volume[i].color = colorscheme.categories[5].color;

    }
}


