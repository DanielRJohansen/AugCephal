#include "VolumeMaker.h"


#include <iostream>
#include <fstream>  
#define LUNG 0
#define FAT 1
#define FLUIDS 2
#define WATER 3
#define MUSCLE 4
#define BLOODCLOT 5
#define HEMATOMA 6
#define BLOOD 7
#define CANCELLOUS 8 
#define CORTICAL 9
#define FOREIGN 10
VolumeMaker::VolumeMaker() {}


VolumeMaker::VolumeMaker(bool default_config) {
    volume = new Block[VOL_X * VOL_Y * VOL_Z];
    loadScans();
    printf("Scans loaded\n");
    CudaOperator CudaOps;
    CudaOps.rotatingMaskFilter(copyVolume(volume), volume);
    CudaOps.kMeansClustering(volume);
    //CudaOps.medianFilter(copyVolume(volume), volume);
   
    // air = -1, unknown = -2
    assignColor();      // And category

    close(CORTICAL);
    close(CANCELLOUS);
    close(MUSCLE);
    //close(FAT);

    if (default_config) {
        cluster(CORTICAL, 500);
        cluster(CANCELLOUS, 100);
        cluster(FOREIGN, 1000);
        cluster(LUNG, 200);
        cluster(FAT, 100);
        cluster(FLUIDS, 20);
        cluster(WATER, 10);
        
        cluster(MUSCLE, 100);
        //cluster(5, 20); //clot
        //cluster(6, 20);
        //cluster(7, 100);
        //cluster(8, 100);
        //cluster(9, 2000);
    }
    
    setIgnore(LUNG, true);  
    setIgnore(FAT, true);
    setIgnore(FLUIDS, true);
    setIgnore(WATER, true);   
    //setIgnore(MUSCLE, true);    
    //setIgnore(BLOODCLOT, true);    
    //setIgnore(HEMATOMA, true);  
    //setIgnore(BLOOD, true);    
    //setIgnore(CANCELLOUS, true);    
    

    locateEmptyYSlices();
    locateEmptyXSlices();
    printf("\n");
    printf("Volume made\n");
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




void VolumeMaker::insertImInVolume(Mat img, int z) { 
    //Mat img_ = cv::Mat::zeros(Size(512, 512), CV_8UC1);
    for (int y = 0; y < VOL_Y; y++) {
        for (int x = 0; x < VOL_X; x++) {
            int hu = img.at<uint16_t>(y, x) - 32768;
            if (hu < HU_MIN || hu > HU_MAX) { 
                volume[xyzToIndex(x, y, z)].ignore = true; 
                //volume[xyzToIndex(x, y, z)].hu_val = OUTSIDE_SPECTRUM;
            }
            //else if (hu > HU_MAX) { volume[xyzToIndex(x, y, z)].ignore = true; }
            //else volume[xyzToIndex(x, y, z)].hu_val = hu;
            volume[xyzToIndex(x, y, z)].hu_val = hu;
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
bool VolumeMaker::isNotClustered(int block_index) { return volume[block_index].cluster_id == NO_CLUSTER; }
bool VolumeMaker::isCategory(int block_index, int cat_id) { return volume[block_index].cat == cat_id; }

void VolumeMaker::cluster(int category_index, int min_cluster_size) {               //Currently not in use
    int cluster_id = 0;
    int clustersize;
    vector<Cluster> clusters;
    printf("\n");
    for (int z = 0; z < VOL_Z; z++) {
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                if (isCategory(block_index, category_index) && isNotClustered(block_index)) {
                    clustersize = propagateCluster(x, y, z, cluster_id, category_index, 1);    // DEEP recursive function
                    Cluster cluster(cluster_id, clustersize);
                    clusters.push_back(cluster);
                    cluster_id++;
                    if (cluster.size >= min_cluster_size) {
                        printf(" \r Clusters of %s : %d ", category_names[category_index].c_str(), --cluster_id);
                    }                    
                }
            }
        }
    }
    for (int i = 0; i < VOL_X * VOL_Y * VOL_Z; i++) {
        if (volume[i].cluster_id != 0 && volume[i].cat == category_index) {
            if (clusters[volume[i].cluster_id].size < min_cluster_size) {
                volume[i].ignore = true;
                volume[i].cat = UNKNOWN_CLUSTER; //Is unknown now;
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






void VolumeMaker::open(int cat_i) {
    // This doesn't work properly yet
    erode(cat_i);
    dilate(cat_i);
    updatePreviousCat();
}
void VolumeMaker::close(int cat_i) {
    dilate(cat_i);
    erode(cat_i);
    updatePreviousCat();
}
void VolumeMaker::dilate(int cat_i) {
    for (int z = 0; z < VOL_Z; z++) {
        //printf(" \r Dilating %s %d", colorscheme.category_ids[cat_i].c_str(), z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);
                //printf("%d     %d", volume[block_index].cat, cat_i);
                if (volume[block_index].cat == cat_i) {
                    for (int z_ = z-1; z_ < z+2; z_++) {
                        for (int y_ = y-1; y_ < y+2; y_++) {
                            for (int x_ = x-1; x_ < x+2; x_++) {
                                if (isLegal(x_, y_, z_)) {
                                    int block_index_ = xyzToIndex(x_, y_, z_);
                                    volume[block_index_].cat_ = cat_i;
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
    for (int z = 0; z < VOL_Z; z++) {
        //printf(" \r Eroding %s %d", colorscheme.category_ids[cat_i].c_str(), z);
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int block_index = xyzToIndex(x, y, z);

                if (volume[block_index].cat_ == cat_i) {
                    for (int z_ = z-1; z_ < z+2; z_++) {
                        for (int y_ = y-1; y_ < y+2; y_++) {
                            for (int x_ = x-1; x_ < x+2; x_++) {

                                if (isLegal(x_, y_, z_)) {
                                    int block_index_ = xyzToIndex(x_, y_, z_);
                                    if (volume[block_index_].cat_ != cat_i) {
                                        volume[block_index].cat_ = volume[block_index].cat;
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
        if (volume[i].cat != volume[i].cat_) {
            volume[i].cat = volume[i].cat_;
            volume[i].color = colormaker.forceColorFromCat(volume[i].cat_, volume[i].hu_val);
            volume[i].ignore = false;
        }
    }
}


bool VolumeMaker::setIgnore(int cat_index, bool hide) {
    if (ignores[cat_index] == hide) {   // Check if anything is changed
        cout << "no change" << endl;
        return false;
    }
    ignores[cat_index] = hide;
    //printf("Ignoring category %d \n", cat_index);
    for (int i = 0; i < VOL_Z * VOL_Y * VOL_X; i++) {
        //if (volume[i].cat_index == cat_index) {
        if (volume[i].cat == cat_index) {
            volume[i].ignore = hide;
        }       
    }
    return true;
}

/* //OBSOLETE//
void VolumeMaker::assignColorFromCat() {
    for (int i = 0; i < VOL_Z * VOL_Y * VOL_X; i++) {
        if (volume[i].ignore)  //This is air
            continue;
        //cout << i << " ";
        volume[i].color = colorscheme.categories[volume[i].cat_index].color;
        //volume[i].color = colorscheme.categories[5].color;
    }
}*/


void VolumeMaker::assignColor() {
    for (int i = 0; i < VOL_Z * VOL_Y * VOL_X; i++) {
        if (volume[i].ignore)  //This is air
            continue;
        volume[i].color = colormaker.colorFromHu(volume[i].hu_val);
        //printf("%f        %f           %f\n", volume[i].color.r, volume[i].color.g, volume[i].color.b);
        volume[i].cat = colormaker.catFromHu(volume[i].hu_val);
        volume[i].cat_ = colormaker.catFromHu(volume[i].hu_val);
        if (volume[i].cat == UNKNOWN_CAT) { volume[i].ignore = true; }
        //volume[i].alpha *= colormaker.belongingFromHu(volume[i].hu_val);
    }
}
void VolumeMaker::assignUnknowns(int unknown_id) {
    for (int z = 0; z < VOL_Z; z++) {
        for (int y = 0; y < VOL_Y; y++) {
            for (int x = 0; x < VOL_X; x++) {
                int i = xyzToIndex(x, y, z);
                int hits = 0;
                int ignores = 0;
                Color c;
                if (!volume[i].ignore && volume[i].cat == unknown_id) {
                    for (int z_ = z - 1; z_ < z + 2; z_++) {
                        for (int y_ = y - 1; y_ < y + 2; y_++) {
                            for (int x_ = x - 1; x_ < x + 2; x_++) {
                                if (isLegal(x_, y_, z_)) {
                                    int i_ = xyzToIndex(x_, y_, z_);
                                    Block b = volume[i_];
                                    if (b.ignore || b.cat == unknown_id) { ignores++; }
                                    else {
                                        c = c + b.color;    // TODO: Fix this lazy solution, add some color dist to equation
                                        hits++;
                                    }
                                }
                            }
                        }
                    }
                    if (ignores > 24) { volume[i].ignore = true; }
                    else { volume[i].color = c * (1. / hits); }     // No logic for chaing cat
                }
            }
        }
    }
}

void VolumeMaker::locateEmptyYSlices() {
    empty_y_slices = new bool[VOL_Y];
    for (int y = 0; y < VOL_Y; y++) {
        empty_y_slices[y] = ySliceIsEmpty(y);
    }
}
void VolumeMaker::locateEmptyXSlices() {
    empty_x_slices = new bool[VOL_X];
    for (int x = 0; x < VOL_X; x++) {
        empty_x_slices[x] = xSliceIsEmpty(x);
    }
}
bool VolumeMaker::ySliceIsEmpty(int y) {
    int non_ignores = 0;
    for (int x = 0; x < VOL_X; x++) {
        for (int z = 0; z < VOL_Z; z++) {
            int block_index = xyzToIndex(x, y, z);
            if (!volume[block_index].ignore) {
                non_ignores++;
                if (non_ignores > NON_IGNORES_THRESHOLD) {
                    return false;
                }
            }
        }
    }
    return true;
}
bool VolumeMaker::xSliceIsEmpty(int x) {
    int non_ignores = 0;
    for (int y = 0; y < VOL_Y; y++) {
        for (int z = 0; z < VOL_Z; z++) {
            int block_index = xyzToIndex(x, y, z);
            if (!volume[block_index].ignore) {
                non_ignores++;
                if (non_ignores > NON_IGNORES_THRESHOLD) {
                    return false;
                }
            }
        }
    }
    return true;
}


void saveNormIm(Mat im, int number, string foldername) {
    float norm_key = 1. / (HU_MAX - HU_MIN);
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
            //outfile << to_string(a) + ' ';
        }
    }
    imwrite("E:\\NormImages\\" + foldername + "\\" + to_string(number) + ".png", im);
}
void VolumeMaker::loadScans() {
    stringvec v;
    read_directory(folder_path, v);
    for (int i = 2; i < VOL_Z + 2; i++) {
        string im_path = folder_path;
        im_path.append(v[i]);
        cout << '\r' << im_path;

        Mat img = imread(im_path, cv::IMREAD_UNCHANGED);
        int z = VOL_Z - 1 - i + 2;
        if (img.empty()) {
            cout << "imload failed" << endl;
            return;
        }
        //saveNormIm(img, i - 2, "163slices");
        insertImInVolume(img, z);
    }

    cout << endl;
    outfile.close();
}