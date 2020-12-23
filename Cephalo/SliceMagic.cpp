#include "SliceMagic.h"
void onMouse(int event, int x, int y, int, void*);



float* global_hu_vals;

SliceMagic::SliceMagic() {
	original = new float[size*size];
    loadOriginal();

    int k = 12;
    int min_n = 2;

    global_hu_vals = copySlice(original);

    float* slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    rotatingMaskFilter(slice, 14);
    showSlice(colorConvert(slice), "RMF");

    slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    rotatingMaskFilter(slice, 14);
    Pixel* image = new Pixel[sizesq];
    sliceToImage(slice, image);
    applyCanny(slice);
    applyEdges(slice, image);
    showSlice(colorConvert(slice), "Canny");


    int num_clusters = cluster(image);
    showImage(image, "Clustered");


    mergeClusters(image, num_clusters, 0.2, 0.09);
    showImage(image, "Merged");
    waitKey();
    //deNoiser(slice);




    waitKey();


}

/*float cm1[25] = { 1, 0.2, 0, 0, 0,  0.2, 1, 0.2, 0, 0,  0, 0.2, 1, 0.2, 0,  0, 0, 0.2, 1, 0.2,  0, 0, 0, 0.2, 1 };
float cm2[25] = { 0, 0, 0, 0.2, 1,  0, 0, 0.2, 1, 0.2,  0, 0.2, 1, 0.2, 0,  0.2, 1, 0.2, 0, 0,  1, 0.2, 0, 0, 0 };
float cm3[25] = { 0, 0, 0, 0, 0,  0.2, 0.2, 0.2, 0.2, 0.2,  1, 1, 1, 1, 1,  0.2, 0.2, 0.2, 0.2, 0.2,  0, 0, 0, 0, 0 };
float cm4[25] = { 0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0 };*/
float cm1[25] = { 1, 0, 0, 0, 0,  0, 1, 0, 0, 0,  0, 0, 1, 0, 0,  0, 0, 0, 1, 0,  0, 0, 0, 0, 1 };
float cm2[25] = { 0, 0, 0, 0, 1,  0, 0, 0, 1, 0,  0, 0, 1, 0, 0,  0, 1, 0, 0, 0,  1, 0, 0, 0, 0 };
float cm3[25] = { 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  1, 1, 1, 1, 1,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0 };
float cm4[25] = { 0, 0, 1, 0, 0,  0, 0, 1, 0, 0,  0, 0, 1, 0, 0,  0, 0, 1, 0, 0,  0, 0, 1, 0, 0 };
float cm5[25] = { 0, 0, 0, 0, 0,  0, 1, 2, 1, 0,  0, 2, 4, 2, 0,  0, 1, 2, 1, 0,  0, 0, 0, 0, 0 };

void copyKernel(float* ori, float* copy, int length) {
    for (int i = 0; i < length; i++) {
        copy[i] = ori[i];
    }
}
void SliceMagic::rotatingMaskFilter(float* slice, int num_masks) {
    Mask masks[14];
    float* copy = copySlice(slice);
    int i = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            masks[i] = Mask(x, y);
            i++;
        }
    }
    masks[9]  = Mask(cm1);   // 4 instead of 5, as a small 20% penalty
    masks[10] = Mask(cm2);
    masks[11] = Mask(cm3);
    masks[12] = Mask(cm4);
    masks[13] = Mask(cm5);
    for (int y = 2; y < size - 2; y++) {
        for (int x = 2; x < size - 2; x++) {
            //if (slice[xyToIndex(x, y)] < -700 || slice[xyToIndex(x, y)] > 700)       // as to not erase bone or brigthen air
            //    continue;
            if (slice[xyToIndex(x, y)] == 0)
                continue;
            // Generate kernel
            float kernel[25];
            int i = 0;
            for (int y_ = y-2; y_ <= y+2; y_++) {
                for (int x_ = x-2; x_ <= x+2; x_++) {
                    kernel[i] = copy[xyToIndex(x_, y_)];
                    i++;
                }
            }

            float best_mean = 0;
            float lowest_var = 9999999;
            float* kernel_copy = new float[25];
            for (int i = 0; i < num_masks; i++) {
                copyKernel(kernel, kernel_copy, 25);
                
                float mean = masks[i].applyMask(kernel_copy);
                float var = masks[i].calcVar(kernel_copy, mean);
                if (var < lowest_var) {
                    lowest_var = var;
                    best_mean = mean;
                }
            }
            slice[xyToIndex(x, y)] = best_mean;
        }
    }

}

inline int radToIndex(float rad) {
    return round((rad + 3.1415) / (2 * 3.1415) * 8);
}
inline bool isLegal(int x, int y) { return x >= 0 && y >= 0 && x < 512 && y < 512; }

void SliceMagic::nonMSStep(vector<int>* edge_indexes, float* steepest_edge, int* steepest_index, float* G, float* theta, bool* fb, int inc, int x, int y, int x_step, int y_step, int step_index, float threshold){
    for (int i = inc; i < size; i += inc) {
        int x_ = x + x_step * i;
        int y_ = y + y_step * i;
        int pp_index = xyToIndex(x_, y_);
        if (!isLegal(x_, y_)) { return; }

        float grad = abs(G[pp_index]);
        int step_index_ = radToIndex(theta[xyToIndex(x_, y_)]);
        //if (grad > threshold) {
        if (grad > min_val && step_index == step_index_) {
            edge_indexes->push_back(pp_index);
            if (grad > * steepest_edge) {
                *steepest_edge = grad;
                *steepest_index = pp_index;
            }
        }
        //else if (abs(i > 1)){ return; }
        else return;
    }
}


void SliceMagic::deNoiser(float* slice) {
    for (int y = 1; y < size-1; y++) {
        for (int x = 1; x < size-1; x++) {
            int p_index = xyToIndex(x, y);
            if (slice[p_index] < normVal(-200, -500, 1000)) {
                slice[p_index] = 1;
                float sum = 0;
                for (int y_ = y - 1; y_ <= y + 1; y_++) {
                    for (int x_ = x - 1; x_ <= x + 1; x_++) {
                        if (!(x_ == x && y_ == y))
                            sum += slice[xyToIndex(x_, y_)];
                    }
                }
                slice[p_index] = sum / 8.;       // Median filter might be better, but im too tired RN.
            }
        }
    }
    
}

void SliceMagic::nonMS(float* slice, float* G, float* theta, bool* forced_black, float threshold) {
    int x_steps[9] = { 1, 1, 0, -1, -1, -1, 0, 1, 1 };
    int y_steps[9] = { 0, 1, 1, 1, 0, -1, -1, -1, 0 };

    vector<int>* edge_indexes = new vector<int>;
    float* steepest_edge = new float;
    int* steepest_index = new int;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int p_index = xyToIndex(x, y);
            if (G[p_index] < threshold) { continue; }

            int step_index = radToIndex(theta[xyToIndex(x, y)]);
            int x_step = x_steps[step_index];
            int y_step = y_steps[step_index];


            edge_indexes->push_back(p_index); 
            *steepest_edge = G[p_index];
            *steepest_index = p_index;  

            nonMSStep(edge_indexes, steepest_edge, steepest_index, G, theta, forced_black, 1, x, y, x_step, y_step, step_index, threshold);
            nonMSStep(edge_indexes, steepest_edge, steepest_index, G, theta, forced_black, -1, x, y, x_step, y_step, step_index, threshold);

            while (!edge_indexes->empty()) {
                int pp_index = edge_indexes->back();
                edge_indexes->pop_back();
                if (!(pp_index == *steepest_index)) {
                    slice[pp_index] = 0;
                    forced_black[pp_index] = true;
                }
            }
        }
    }
    delete(edge_indexes);
    delete(steepest_edge);
    delete(steepest_index);
}


void SliceMagic::propagateHysteresis(float* slice, bool* forced_black, float* G, int x, int y) {
    slice[xyToIndex(x, y)] = 1;
    float highest = 0;
    int2 best;
    for (int y_ = y - 1; y_ <= y + 1; y_++) {
        for (int x_ = x - 1; x_ <= x + 1; x_++) {
            
            if (!isLegal(x_, y_))
                continue;
            int index = xyToIndex(x_, y_);
            
            if (G[index] < grad_threshold && G[index] > min_val && G[index] > 0 && slice[index] == 0){//&& forced_black[xyToIndex(x, y)]) { //Dont think the last one matters, due to the first one
                highest = G[index];
                best = int2(x_, y_);
                propagateHysteresis(slice, forced_black, G, best.x, best.y);
            }
            
        }
    }
    //if (highest > 0)
     //   propagateHysteresis(slice, forced_black, G, best.x, best.y);
}

void SliceMagic::hysteresis(float* slice, float* G, bool* forced_black) {
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int index = xyToIndex(x, y);

            if (G[index] > grad_threshold && !forced_black[index]) {
                propagateHysteresis(slice, forced_black, G, x, y);
            }
            
        }
    }
}

void SliceMagic::applyCanny(float* slice) {
    
    float Maskver[9] = { -3, -10, -3, 0, 0, 0, 3, 10, 3 };
    float Maskhor[9] = { -3, 0, 3, -10, 0, 10, -3, 0, 3 };

    float* gradx = new float[sizesq];
    float* grady = new float[sizesq];
    float* G = new float[sizesq];
    float* theta = new float[sizesq];


   
    float* kernel = new float[9];
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            if (y * x == 0 || y == size - 1 || x == size - 1) {
                G[xyToIndex(x, y)] = 0;
                theta[xyToIndex(x, y)] = 0;
                continue;
            }
                

            float Gx = 0;
            float Gy = 0;
            int i = 0; 
            for (int y_ = y - 1; y_ <= y + 1; y_++) {
                for (int x_ = x - 1; x_ <= x + 1; x_++) {
                    kernel[i] = slice[xyToIndex(x_, y_)];
                    Gx += (kernel[i] * Maskhor[i]);
                    Gy += (kernel[i] * Maskver[i]);
                    i++;
                }
            }
            G[xyToIndex(x,y)] = sqrt(Gx*Gx + Gy*Gy);
            float a = Gy / Gx;
            theta[xyToIndex(x, y)] = atan2(Gy, Gx);
        }
    }
    float largest_val = 0;
    for (int i = 0; i < sizesq; i++) {
        if (G[i] > largest_val)
            largest_val = G[i];
    }
    for (int i = 0; i < sizesq; i++) {
            G[i] /= largest_val;
    }
    for (int i = 0; i < sizesq; i++) {
        if (G[i] > grad_threshold)
            slice[i] = 1;
        else
            slice[i] = 0;
    }
    bool* forced_black = new bool[sizesq];
    for (int i = 0; i < sizesq; i++) {forced_black[i] = 0; }
    nonMS(slice, G, theta, forced_black, grad_threshold);
    hysteresis(slice, G, forced_black);
    nonMS(slice, G, theta, forced_black, min_val);

}


int x_off[4] = { 0, 0, -1, 1 };
int y_off[4] = { -1, 1, 0, 0 };
void SliceMagic::propagateCluster(Pixel* image, int cluster_id, Color3 color, float* acc_mean, int* n_members, int* member_indexes, int2 pos) {
    int index = xyToIndex(pos);
    image[index].reserve();

    //printf("%d\n", *n_members);
    member_indexes[*n_members] = index;
    *n_members += 1;
    *acc_mean += image[index].getVal();
    
    for (int i = 0; i < 4; i++) {
        int x_ = pos.x + x_off[i];
        int y_ = pos.y + y_off[i];
        
        if (!isLegal(x_, y_))
            continue;

        int index_ = xyToIndex(x_, y_);
        if (!image[index_].isReserved() && !image[index_].isEdge())
            propagateCluster(image, cluster_id, color, acc_mean, n_members, member_indexes, int2(x_, y_));
    }

}

int SliceMagic::cluster(Pixel* image) {
    int id = 0;
    Color3 color(rand() % 255, rand() % 255, rand() % 255);
    float* acc_mean = new float(0);
    int* n_members = new int(0);
    int* member_indexes = new int[sizesq];
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int index = xyToIndex(x, y);

            if (!image[index].isReserved() && !image[index].isEdge()) {
                propagateCluster(image, id, color, acc_mean, n_members, member_indexes, int2(x,y));

                // Do for all members in cluster
                float cluster_mean = *acc_mean / (float) *n_members;
                for (int i = 0; i < *n_members; i++) {
                    image[member_indexes[i]].assignToCluster(id, color, cluster_mean);
                }

                // Prepare for next cluster;
                id++;
                color = Color3(rand() % 255, rand() % 255, rand() % 255);
                *n_members = 0; // We dont need to overwrite the member list, as we only read untill n_mem, rest is overwritten
                *acc_mean = 0;
            }
        }
    }
    delete(acc_mean, n_members, member_indexes);

    for (int i = 0; i < sizesq; i++) { 
        image[i].checkAssignBelonging(image); 
    }  // IDEA: do multiple times, to get rid of edge surrounded by edges.
    printf("%d clusters found \n", id);
    return id;  // Num clusters
}

void makeClusterSublist(TissueCluster* original, TissueCluster** sub, int* indexes, int size){   // Verified it works
    for (int i = 0; i < size; i++) {
        sub[i] = &original[indexes[i]];
    }
}
void SliceMagic::mergeClusters(Pixel* image, int num_clusters, float max_absolute_dist, float max_fractional_dist) {
    printf("Merging clusters. RAM req: %d Kb\n", num_clusters * sizeof(TissueCluster) / 1000);
    TissueCluster* TC = new TissueCluster[num_clusters];
    for (int i = 0; i < sizesq; i++) {
        int c_index = image[i].getID();
        if (c_index != -1) {    // We dont want edges
            if (!TC[c_index].initialized)
                TC[c_index] = TissueCluster(image[i]);
            else
                TC[c_index].addToCluster(image[i]);
        }
    }

    for (int i = 0; i < sizesq; i++) {
        if (!image[i].isEdge())
            continue;
        int* connected_indexes = new int[9];
        int num_connected = image[i].connectedClusters(image, connected_indexes);
        if (num_connected > 0) {
            int survivor_index = connected_indexes[0];
            TissueCluster survivor_copy = TC[survivor_index];

            if (num_connected == 1) {
                image[i].assignToCluster(survivor_copy.cluster_id, survivor_copy.color, survivor_copy.cluster_mean);
                TC[connected_indexes[0]].addToCluster(image[i]);
            }
            else {
                // The first cluster (at index 0) will be only surviving cluster, others deathmarked. 
                // All pixels will belong to first cluster;
                TissueCluster** sublist = new TissueCluster * [num_connected];
                makeClusterSublist(TC, sublist, connected_indexes, num_connected);

                bool mergeable = TC[survivor_index].isMergeable(sublist, num_connected, max_absolute_dist, max_fractional_dist);
                if (mergeable) {
                    TC[survivor_index].mergeClusters(sublist, image, num_connected);
                    TC[survivor_index].addToCluster(image[i]);
                    image[i].assignToCluster(TC[survivor_index].cluster_id, TC[survivor_index].color, TC[survivor_index].cluster_mean);
                }
                delete(sublist);
            }
        }
        
        else
            image[i].color = Color3(0, 0, 0);
        delete(connected_indexes);
    }
    delete(TC);
}


void SliceMagic::findNeighbors(Pixel* image) {
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            for (int y_ = y - 1; y_ <= y + 1; y_++) {
                for (int x_ = x - 1; x_ <= x + 1; x_++) {
                    if (isLegal(x_, y_))
                        image[xyToIndex(x, y)].addNeighbor(xyToIndex(x_, y_));
                }
            }
        }
    }
}

void SliceMagic::kMeans(float* slice, int k, int iterations) {
    Kcluster* clusters = new Kcluster[k];
    for (int i = 0; i < k; i++) {
        clusters[i] = Kcluster((float)i/k);
    }

    //Iterate clustering
    for (int iter = 0; iter < iterations; iter++) {
        for (int p = 0; p < sizesq; p++) {
            float best = 0; 
            int best_index = 0;
            for (int i = 0; i < k; i++) {
                float score = clusters[i].belonging(slice[p]);
                if (score > best) {
                    best = score;
                    best_index = i;
                }
            }
            if (iter == 0) { clusters[p%k].addMember(slice[p]); } // Adds basically random value to cluster -- LOL I DO THIS 20 TIMES, BASICALLY THIS IS 1 ITERATION SAD!!!!
            else { clusters[best_index].addMember(slice[p]); }
            
        }

        for (int i = 0; i < k; i++) {
            clusters[i].updateCluster();
        }
    }

    // Assign color from best cluster
    for (int p = 0; p < sizesq; p++) {
        float best = 0;
        int best_index = 0;
        for (int i = 0; i < k; i++) {
            float score = clusters[i].belonging(slice[p]);
            if (score > best) {
                best = score;
                best_index = i;
            }
        }
        slice[p] = clusters[best_index].assigned_val;
    }
}

float* SliceMagic::copySlice(float* slice) {
    float* copy = new float[size * size];
    for (int i = 0; i < sizesq; i++) {
        copy[i] = slice[i];
    }
    return copy;
}

void SliceMagic::loadOriginal() {
	string im_path = "E:\\NIH_images\\002701_04_03\\160.png";
	Mat img = imread(im_path, cv::IMREAD_UNCHANGED);

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int hu = img.at<uint16_t>(y, x) - 32768;
            original[xyToIndex(x, y)] = hu;
        }
    }
}


void SliceMagic::windowSlice(float* slice, float min, float max) {
    for (int i = 0; i < size * size; i++) {
        float hu = slice[i];
        if (hu > max) { slice[i] = 1; }
        else if (hu < min) { slice[i] =0; }
        else { slice[i] = normVal(hu, min, max); }
    }
}

Color3* SliceMagic::colorConvert(float* slice) {
    Color3* slice_ = new Color3[sizesq];
    for (int i = 0; i < sizesq; i++) {
        slice_[i] = slice[i];
    }
    return slice_;
}

float SliceMagic::median(float* window) {
    float sorted[9];
    for (int i = 0; i < 9; i++) {
        float lowest = 99999;
        int lowest_index = 0;
        for (int j = 0; j < 9; j++) {
            if (window[j] < lowest) {
                lowest = window[j];
                lowest_index = j;
            }
        }
        sorted[i] = window[lowest_index];
        window[lowest_index = 99999];
    }
    return sorted[4];
}

void SliceMagic::medianFilter(float* slice) {
    float* copy = copySlice(slice);
    float* window = new float[9];

    for (int y = 1; y < size - 1; y++) {
        for (int x = 1; x < size - 1; x++) {

            int i = 0;
            for (int y_ = y-1; y_ < y + 2; y_++) {
                for (int x_ = x - 1; x_ < x + 2; x_++) {
                    window[i] = copy[xyToIndex(x_, y_)];
                    i++;
                }
            }
            slice[xyToIndex(x, y)] = median(window);
            //cout << median(window);
        }
    }
    delete(window);
    delete(copy);
}

void SliceMagic::showSlice(Color3* slice, string title) {
    Mat img(size, size, CV_8UC3);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            Color3 c = slice[xyToIndex(x, y)] * 255.;
            img.at<Vec3b>(y, x)[0] = c.r;
            img.at<Vec3b>(y, x)[1] = c.g;
            img.at<Vec3b>(y, x)[2] = c.b;
        }
    }
    //namedWindow(title, WINDOW_NORMAL);
    imshow(title, img);
    setMouseCallback(title, onMouse, 0);
}

void SliceMagic::showImage(Pixel* image, string title) {
    Mat img(size, size, CV_8UC3);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            Color3 c = image[xyToIndex(x, y)].color;
            img.at<Vec3b>(y, x)[0] = c.r;
            img.at<Vec3b>(y, x)[1] = c.g;
            img.at<Vec3b>(y, x)[2] = c.b;
        }
    }
    //namedWindow(title, WINDOW_NORMAL);
    imshow(title, img);
    setMouseCallback(title, onMouse, 0);
}

struct ClusterThingy {
    float val = -1;
    int count = 0;
};

void SliceMagic::assignToMostCommonNeighbor(float* slice, int x, int y) {
    ClusterThingy* CT;
    
    for (int y_ = y - 1; y_ < y + 2; y_++) {
        for (int x_ = x - 1; x_ < x + 2; x_++) {
            CT = new ClusterThingy[9];
            float val = slice[xyToIndex(x_, y_)];


            for (int i = 0; i < 9; i++) {
                if (CT[i].val == -1) {
                    CT[i].val = val;
                    CT[i].count++;
                }
                else if (CT[i].val == val)
                    CT[i].count++;
            }


            int best_neighbor = 0;
            int best_index = 0;
            for (int i = 0; i < 9; i++) {
                if (CT[i].count > best_neighbor) {
                    best_neighbor = CT[i].count;
                    best_index = i;
                }
            }
            slice[xyToIndex(x, y)] = CT[best_index].val;
            delete(CT);
        }
    }
}



void SliceMagic::requireMinNeighbors(float* slice, int min) {
    for (int y = 1; y < size - 1; y++) {
        for (int x = 1; x < size - 1; x++) {
            float pval = slice[xyToIndex(x, y)];
            
            int neighbors = 0;

            for (int y_ = y - 1; y_ < y + 2; y_++) {
                for (int x_ = x - 1; x_ < x + 2; x_++) {
                    if (y == y_ && x == x_)
                        continue;
                    //cout << slice[xyToIndex(x_, y_)] << "       " << pval << endl;
                    if (slice[xyToIndex(x_, y_)] == pval)
                        neighbors++;
                }
            }
            if (neighbors < min)
                assignToMostCommonNeighbor(slice, x, y);
        }
    }
}





void onMouse(int event, int x, int y, int, void*)
{


    Point pt = Point(x, y);
    std::cout << "(" << pt.x << ", " << pt.y << ")      huval: "<< global_hu_vals[y*512+x] << '\n';

}


void Pixel::checkAssignBelonging(Pixel* image) {
    if (!is_edge) return;
    int cluster_ = -1;
    Color3 color_;
    float mean_;
    for (int i = 0; i < n_neighbors; i++) {
        Pixel neighbor = image[neighbor_indexes[i]];

        if (cluster_ == -1) {
            cluster_ = neighbor.cluster_id;
            color_ = neighbor.color;
            mean_ = neighbor.getClusterMean();
        }
        else if (neighbor.cluster_id == cluster_ || neighbor.is_edge) {}
        else return;
    }
    if (cluster_ != -1) {// Else do nothing to this pixel
        assignToCluster(cluster_, color_, mean_);
    }
}
bool isInList(int* list, int size, int target) {
    for (int i = 0; i < size; i++) {
        if (list[i] == target)
            return true;
    }
    return false;
}
int Pixel::connectedClusters(Pixel* image, int* connected_indexes) {
    int num_clusters = 0;
    for (int i = 0; i < n_neighbors; i++) {
        int id_ = image[neighbor_indexes[i]].cluster_id;
        if (id_ == -1)
            continue;
        //printf("%d  ", id_);

        if (!isInList(connected_indexes, num_clusters, id_)) {
            connected_indexes[num_clusters] = id_;
            num_clusters++;
        }
        
    }
    //printf("\n");
    return num_clusters;
}






TissueCluster::TissueCluster(Pixel p) {
    min_val = p.getClusterMean();
    max_val = p.getClusterMean();
    //printf("Initting cluster from mean: %f\n", p.cluster_mean);
    cluster_mean = p.getClusterMean();
    color = p.color;
    initialized = true;
    cluster_id = p.cluster_id;
    //printf("Initializing cluster %d\n", cluster_id);
    addToCluster(p);
}



void getMinMax(TissueCluster** clusters, int num_clusters, float* min, float* max) {
    *min = 99999999;
    *max = -9999999;
    for (int i = 0; i < num_clusters; i++) {
        float cmin = clusters[i]->getMin();
        float cmax = clusters[i]->getMax();
        //printf("cmin cmax    %f     %f\n", cmin, cmax);
        if (cmin < *min)
            *min = cmin;
        if (cmax > *max)
            *max = cmax;
    }
    if (*min > *max)
        printf("WTF WENT WRONG HERE, MIN LARGER THAN MAX\n");
}
bool TissueCluster::isMergeable(TissueCluster** clusters, int num_clusters, float absolute_dif, float relative_dif) {
    float* min = new float;
    float* max = new float;
    getMinMax(clusters, num_clusters, min, max);

    float abs_dif = *max - *min;
    float rel_dif = *max / *min;
    //printf("abs   rel   %f       %f\n", abs_dif, rel_dif);
    //printf("min max    %f    %f\n\n", *min, *max);
    delete(min, max);
    if (abs_dif < absolute_dif || rel_dif < relative_dif)
        return true;
    return false;
}

void TissueCluster::mergeClusters(TissueCluster** clusters, Pixel* image, int num_clusters) {

    // This is done so cluster cant be moved gradually - the most extreme means will be kept.
    float* min = new float;
    float* max = new float;
    getMinMax(clusters, num_clusters, min, max);
    min_val = *min;
    max_val = *max;
    delete(min, max);

    // First handle the metadata of the survivor cluster
    int total_size = 0;
    int acc_mean = 0;
    for (int i = 0; i < num_clusters; i++) {
        total_size += clusters[i]->getSize();
        acc_mean += clusters[i]->cluster_mean * clusters[i]->getSize();
    }
    //clusters[0].cluster_size = total_size;    DO NOT CHANGE SIZE HERE! THE SIZE IS USED TO ADD NEW PIXELS BELOW!!
    clusters[0]->cluster_mean = acc_mean / total_size;

    // Second redistribute each pixel to the survivor cluster
    for (int i = 1; i < num_clusters; i++) {
        for (int j = 0; j < clusters[i]->cluster_size; j++) {
            // For each pixel j in non-survivor cluster i
            int pixel_index = clusters[i]->getPixel(j);
            //
            //printf("cluster %d  pixel %d    pixel_index %d \n", i, j, pixel_index);
            //printf("Transferring pixel: %d\n", pixel_index);
            clusters[0]->addToCluster(image[pixel_index]);
            image[pixel_index].assignToCluster(clusters[0]->cluster_id, clusters[0]->color, clusters[0]->cluster_mean);
        }
        clusters[i]->deadmark();
    }
}

void copyPtr(int* from, int* to, int size) {
    for (int i = 0; i < size; i++) {
        to[i] = from[i];
    }
}
void TissueCluster::handleArraySize() {
    if (cluster_size == allocated_size) {
        int new_size;
        if (allocated_size == 0)
            new_size = 100;
        else if (allocated_size == 100)
            new_size = 32768;
        else if (allocated_size == 32768)
            new_size = 524288;
        else
            new_size = allocated_size * 2;

        int* copy = pixel_indexes;
        pixel_indexes = new int[new_size];
        copyPtr(copy, pixel_indexes, allocated_size);
        allocated_size = new_size;
    }
}

void TissueCluster::addToCluster(Pixel pixel) {
    handleArraySize();
    pixel_indexes[cluster_size] = pixel.index;
    cluster_size++;
}