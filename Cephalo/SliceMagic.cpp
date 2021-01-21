#include "SliceMagic.h"
#include "Resizer.h"

void onMouse(int event, int x, int y, int, void*);



float* global_hu_vals;
float* global_km_vals;
float* global_cat_vals;
Pixel* global_im;
TissueCluster* global_clusters;

int* bucketSort(int* sizes, int num) {	// Fucks up on size 0 or less!
    int* ordered_indexes = new int[num];
    int head = num - 1;

    int bucket_start = 0;
    int bucket_end = 4;
    while (true) {
        for (int i = 0; i < num; i++) {
            int sizei = sizes[i];
            if (sizei >= bucket_start && sizei < bucket_end) {
                ordered_indexes[head] = i;
                head--;
                if (head == -1)
                    return ordered_indexes;
            }
        }
        bucket_start = bucket_end;
        bucket_end *= 2;
        if (bucket_end == 0) {
            printf("SOMETHING WENT WRONG");
            break;
        }
            
    }
}

float medianOfList(float* list, int size) {
    float* ordered_list = new float[size];
    float ignore = 2;
    for (int i = 0; i < size; i++) {
        if (list[i] > 1) {
            printf("List size %d:\n", size);
            for (int i = 0; i < size; i++) {
                if (i == size / 2)
                    printf(" ----------->");
                printf("    %f     \n", list[i]);
            }
            printf("Median: %f", ordered_list[size / 2]);
            for (int j = 0; j < 1000000000; j++) {
                float f = list[j];
            }
        }
    }
    

    for (int i = 0; i < size; i++) {
        float lowest = ignore;
        int lowest_index = 0;
        for (int j = 0; j < size; j++) {
            if (list[j] < lowest) {
                lowest = list[j];
                lowest_index = j;
            }
        }
        ordered_list[i] = list[lowest_index];
        list[lowest_index] += ignore;
    }

    float median = ordered_list[size / 2];
    if (median > 1) {
        printf("Ordered list size %d:\n", size);
        for (int i = 0; i < size; i++) {
            if (i == size / 2) 
                printf(" ----------->");
            printf("    %f     \n", ordered_list[i]);
        }
        printf("Median: %f", ordered_list[size / 2]);
        for (int j = 0; j < 1000000000; j++) {
            float f = list[j];
        }
    }
    return median;
}

void onMouse(int event, int x, int y, int, void*)
{   
    if (x < 0 || y < 0)
        return;
    Point pt = Point(x, y);
    //std::cout << "(" << pt.x << ", " << pt.y << ")      huval: " << global_hu_vals[y * ss + x] << '\n';
    //Pixel p = global_im[y * ss + x];
    //printf("\r(%d, %d)     Hu: %f     ID: %05d  Size: %07d   Cluster median value: %f", pt.x, pt.y, global_hu_vals[y * ss + x], p.cluster_id, global_clusters[p.cluster_id].getSize(), global_cat_vals[y * ss + x]);

    //printf("(%d, %d)     Hu: %f     Cluster: %f  \n", pt.x, pt.y, global_hu_vals[y * ss + x], global_km_vals[y * ss + x]);
    
    //printf("(%d, %d)     Hu: %f     \n", pt.x, pt.y, global_hu_vals[y * ss + x]);
    
    
}
void setGlobalLookup(TissueCluster* clusters, int num_clusters, Pixel* image, int size) {
    global_cat_vals = new float[size];
    for (int i = 0; i < size; i++) {
        global_cat_vals[i] = image[i].median;
    }
    global_im = new Pixel[size];
    for (int i = 0; i < size; i++) {
        global_im[i] = image[i];
    }
    global_clusters = new TissueCluster[num_clusters];
    for (int i = 0; i < num_clusters; i++) {
        global_clusters[i] = clusters[i];
    }
}
void setGlobalKMLookup(float* slice, int size) {
    global_km_vals = new float[size];
    for (int i = 0; i < size; i++) {
        global_km_vals[i] = slice[i];
    }
}
void makeHistogram(float* slice, int num_bins, int size, int min = 0, int max = 1) {
    float binsize = ((float)max - (float)min) / (float)num_bins;
    int* histogram = new int[num_bins]();

    for (int i = 0; i < size; i++) {
        int index = ceil(slice[i] / binsize);
        histogram[index]++;
    }

    printf("Histogram with %d bins, binsize: %f\n", num_bins, binsize);
    printf("[%f", histogram[0]);
    for (int i = 1; i < num_bins; i++) {
        printf(", %d", histogram[i]);
    }
    printf("];\n");
}

SliceMagic::SliceMagic() {
    int from = -1000;
    int to = 500;

    float* original = loadOriginal();
 
    Resizer resizer(load_size, size);
    
    float* slice = resizer.Interpolate2D(original);
    
    int k = 16;
    int iter = 20;
    int min_n = 2;
    windowSlice(original, from, to, 512);
    showSlice(colorConvert(original, 512 * 512), "Oríginal image", 512);


    
    windowSlice(slice, from, to, 1024);
    showSlice(colorConvert(slice), "Linear window from -700 to 500");
    waitKey();
    return;

    
    rotatingMaskFilter(slice, 14);
    global_hu_vals = copySlice(slice);
    showSlice(colorConvert(slice), "Rotating Mask Filtered");


    int num_clusters;
    Pixel* image2 = new Pixel[sizesq];
    sliceToImage(slice, image2);
    fuzzyMeans(image2, slice, 14);
    TissueCluster* clusters = cluster(image2, &num_clusters, "absolute_values");
    showImage(image2, "Fuzzy Means Clustered");

    int remaining_clusters = orderedPropagatingMerger(clusters, image2, num_clusters, 0.05);
    setGlobalLookup(clusters, num_clusters, image2, sizesq);
    showImage(image2, "Clusters Merged!");

    vesicleElimination(clusters, image2, num_clusters, 10, 80, 0.2, remaining_clusters);
    showImage(image2, "Vesicles eliminated");
    waitKey();

}

/*float cm1[25] = { 1, 0.2, 0, 0, 0,  0.2, 1, 0.2, 0, 0,  0, 0.2, 1, 0.2, 0,  0, 0, 0.2, 1, 0.2,  0, 0, 0, 0.2, 1 };
float cm2[25] = { 0, 0, 0, 0.2, 1,  0, 0, 0.2, 1, 0.2,  0, 0.2, 1, 0.2, 0,  0.2, 1, 0.2, 0, 0,  1, 0.2, 0, 0, 0 };
float cm3[25] = { 0, 0, 0, 0, 0,  0.2, 0.2, 0.2, 0.2, 0.2,  1, 1, 1, 1, 1,  0.2, 0.2, 0.2, 0.2, 0.2,  0, 0, 0, 0, 0 };
float cm4[25] = { 0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0 };*/
int cm1[25] = { 1, 0, 0, 0, 0,  0, 1, 0, 0, 0,  0, 0, 2, 0, 0,  0, 0, 0, 1, 0,  0, 0, 0, 0, 1 };
int cm2[25] = { 0, 0, 0, 0, 1,  0, 0, 0, 1, 0,  0, 0, 2, 0, 0,  0, 1, 0, 0, 0,  1, 0, 0, 0, 0 };
int cm3[25] = { 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,  1, 1, 2, 1, 1,  0, 0, 0, 0, 0,  0, 0, 0, 0, 0 };
int cm4[25] = { 0, 0, 1, 0, 0,  0, 0, 1, 0, 0,  0, 0, 2, 0, 0,  0, 0, 1, 0, 0,  0, 0, 1, 0, 0 };
int cm5[25] = { 0, 0, 0, 0, 0,  0, 1, 2, 1, 0,  0, 2, 4, 2, 0,  0, 1, 2, 1, 0,  0, 0, 0, 0, 0 };

void copyKernel(float* ori, float* copy, int length) {
    for (int i = 0; i < length; i++) {
        copy[i] = ori[i];
    }
}

// THIS SHIT DANGEROUS AS FUCK AS WE DONT JUSTTTTT HAVE 9 EQUALLY WEIGHED VALUES IN THE KERNEL


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
            float lowest_var = 19191919;
            float* kernel_copy = new float[25];
            for (int i = 0; i < num_masks; i++) {
                copyKernel(kernel, kernel_copy, 25);
                
                float mean = masks[i].applyMask(kernel_copy);
                float var = masks[i].calcVar(kernel_copy, mean);
                if (var < lowest_var) {
                    lowest_var = var;
                    best_mean = mean;
                    //best_mean = masks[i].median(kernel);
                }
            }
            //printf("best mean: %f\n", best_mean);
            slice[xyToIndex(x, y)] = best_mean;
            delete(kernel_copy);
        }
    }
    delete(copy);
}

inline int radToIndex(float rad) {
    return round((rad + 3.1415) / (2 * 3.1415) * 8);
}

void SliceMagic::nonMSStep(vector<int>* edge_indexes, float* steepest_edge, int* steepest_index, float* G, float* theta, bool* fb, int inc, int x, int y, int x_step, int y_step, int step_index, float threshold){
    for (int i = inc; i < size; i += inc) {
        int x_ = x + x_step * i;
        int y_ = y + y_step * i;
        int pp_index = xyToIndex(x_, y_);
        if (!isLegal(x_, y_)) { return; }

        float grad = abs(G[pp_index]);
        int step_index_ = radToIndex(theta[xyToIndex(x_, y_)]);
        //if (grad > threshold) {
        if (grad < min_val)     // Only terminate the search once we get below the threshold
            return;
        if (step_index == step_index_) {  // Step index ensures they have same orientation, otherwise continue
            edge_indexes->push_back(pp_index);
            if (grad > *steepest_edge) {
                *steepest_edge = grad;
                *steepest_index = pp_index;
            }
        }
        else return;            // doesn't seem to make any difference
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
    delete(edge_indexes, steepest_edge, steepest_index);
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
    //-nonMS(slice, G, theta, forced_black, grad_threshold);
    hysteresis(slice, G, forced_black);
    nonMS(slice, G, theta, forced_black, min_val);
    delete(forced_black, gradx, grady, G, theta);
}


int x_off[4] = { 0, 0, -1, 1 };
int y_off[4] = { -1, 1, 0, 0 };

void SliceMagic::propagateCluster(Pixel* image, int cluster_id, Color3 color, float* acc_mean, int* n_members, int* member_indexes, int2 pos, string type) {
    int index = xyToIndex(pos);
    image[index].reserve();
    member_indexes[*n_members] = index;
    *n_members += 1;
    *acc_mean += image[index].getVal();
    
    for (int i = 0; i < 4; i++) {
        int x_ = pos.x + x_off[i];
        int y_ = pos.y + y_off[i];
        
        if (!isLegal(x_, y_))
            continue;

        int index_ = xyToIndex(x_, y_);
        if (type == "edge_separation") {
            if (!image[index_].isReserved() && !image[index_].isEdge())
                propagateCluster(image, cluster_id, color, acc_mean, n_members, member_indexes, int2(x_, y_), type);
        }
        else if (type == "absolute_values") {
            if (!image[index_].isReserved() && image[index_].k_cluster == image[index].k_cluster)
                propagateCluster(image, cluster_id, color, acc_mean, n_members, member_indexes, int2(x_, y_), type);
        }
    }
}


TissueCluster* initTissueCluster(Pixel* image, int num_clusters, int sizesq) {
    TissueCluster* clusters = new TissueCluster[num_clusters];
    for (int i = 0; i < num_clusters; i++) {
        clusters[i] = TissueCluster(i, num_clusters);
    }

    for (int i = 0; i < sizesq; i++) {
        Pixel p = image[i];
        clusters[p.cluster_id].addToCluster(p, image);
        image[i].assignToCluster(clusters[p.cluster_id].cluster_id, clusters[p.cluster_id].color);  // Mainly just to give the pixel the correct color.
    }
    return clusters;
}
TissueCluster* SliceMagic::cluster(Pixel* image, int* num_clusters, string type) {
    printf("Clustering using %s\n", type);
    
    int id = 0;
    Color3 color = Color3().getRandColor();
    float* acc_mean = new float(0);
    int* n_members = new int(0);
    int* member_indexes = new int[sizesq];
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int index = xyToIndex(x, y);

            if (type == "edge_separation") {
                if (image[index].isReserved() || image[index].isEdge()) 
                    continue;
                
                propagateCluster(image, id, color, acc_mean, n_members, member_indexes, int2(x, y), type);

            }
            else if (type == "absolute_values") {
                if (image[index].isReserved()) 
                    continue;
                
                propagateCluster(image, id, color, acc_mean, n_members, member_indexes, int2(x, y), type);
            }
            else {
                printf("WRONG CLUSTER TYPE, EXITING");
                return 0;
            }
                
            // Do for all members in cluster
            float cluster_mean = *acc_mean / (float) *n_members;
            for (int i = 0; i < *n_members; i++) {
                image[member_indexes[i]].assignToCluster(id, color);
            }
            // Prepare for next cluster;
            id++;
            color = Color3().getRandColor();
            *n_members = 0; // We dont need to overwrite the member list, as we only read untill n_mem, rest is overwritten
            *acc_mean = 0;          
        }
    }
    delete(acc_mean, n_members, member_indexes);
    *num_clusters = id;
    printf("Calculating cluster medians. ");
    TissueCluster* clusters = initTissueCluster(image, *num_clusters, sizesq);
    for (int i = 0; i < *num_clusters; i++) {   
        clusters[i].recalcCluster(image);                                                                                       
    }
    printf("%d clusters found \n", id);

    return clusters;
}

void SliceMagic::assignClusterMedianToImage(Pixel* image, int num_clusters) {   // I think this is obsolete
    TissueCluster* clusters = initTissueCluster(image, num_clusters, sizesq);

    for (int i = 0; i < num_clusters; i++) {
        clusters[i].recalcCluster(image);
    }
    for (int i = 0; i < sizesq; i++) {
        image[i].median = clusters[image[i].cluster_id].getMedian();
    }
}

int* orderClustersBySize(TissueCluster* clusters, int num_clusters) { 
    int* sizes = new int[num_clusters];
    for (int i = 0; i < num_clusters; i++) {
        sizes[i] = clusters[i].getSize();
    }
    int* ordered_indexes = bucketSort(sizes, num_clusters);
    delete(sizes);
    return ordered_indexes;
}
int SliceMagic::orderedPropagatingMerger(TissueCluster* clusters, Pixel* image, int num_clusters, float max_absolute_dist) {
    int num_merges = 0;
    int *num_mergeables = new int;
    // Find size order
    int* ordered_indexes = orderClustersBySize(clusters, num_clusters);

    for (int i = 0; i < num_clusters; i++) {
        int index =  ordered_indexes[i];
        if (clusters[index].isDeadmarked())
            continue;
        while (true) {
            *num_mergeables = 0;

            TissueCluster** mergeable_clusters = clusters[index].findMergeables(clusters, num_clusters, max_absolute_dist, num_mergeables);
            int merges = clusters[index].mergeClusters(mergeable_clusters, image, *num_mergeables);
            //clusters[index].recalcCluster(image);    // Update median vals 
            //delete(mergeable_clusters); 
            num_merges += merges;


            if (merges == 0)
                break;
        }
    }

    delete(ordered_indexes, num_mergeables);
    printf("\n              Merging finished. Clusters: %05d -> %05d\n\n", num_clusters, num_clusters - num_merges);
    return num_clusters - num_merges;
}

int SliceMagic::vesicleElimination(TissueCluster* clusters, Pixel* image, int num_clusters, int size1, int size2, float size2_threshold, int num_remaining_clusters) {
    int eliminations = 0;
    for (int i = 0; i < num_clusters; i++) {
        if (clusters[i].isDeadmarked())
            continue;
        if (clusters[i].getSize() < size2) {
            if (clusters[i].getSize() < size1) {
                clusters[i].assignToClosestNeighbor(clusters, image, num_clusters);
                eliminations++;
            }
            else {
                if (clusters[i].getNumLiveNeighbors(clusters, num_clusters) < 3) {
                    clusters[i].assignToClosestNeighbor(clusters, image, num_clusters, size2_threshold);
                    eliminations++;
                }
            }
        }

    }
    int remaining = num_remaining_clusters - eliminations;
    printf("\n              Vesicle elimination finished. %d -> %d\n\n", num_remaining_clusters, remaining);
    return remaining;
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

Kcluster* SliceMagic::kMeans(float* slice, int k, int iterations) {
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
            if (iter == 0) { clusters[p%k].addMember(slice[p]); } // Adds basically random value to cluster
            else { clusters[best_index].addMember(slice[p]); }
            
        }
        float total_change = 0;
        for (int i = 0; i < k; i++) {
            total_change += clusters[i].updateCluster();
        }
        printf("\rPerforming kmeans iteration %d of %d. Change: %f  ", iter, iterations, total_change);
        if (total_change < 0.02)
            break;
    }
    printf("\nK-Clustermeans:\n");
    for (int i = 0; i < k; i++) {
        printf("    Cluster %d.   Members: %d   Mean: %f  \n", i, clusters[i].prev_members, clusters[i].centroid);
    }
    printf("\n");
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
        slice[p] = clusters[best_index].centroid;
    }
    printf("\n");
    return clusters;
}

//float gauss_kernel[25] = { 1, 4, 7, 4, 1,    4, 16, 26, 16, 4,    7, 26, 41, 26, 7,   4, 16, 26, 16, 4,     1, 4, 7, 4, 1 };
float gauss_kernel[9] = { 1/16.,1/8.,1/16.,     1/8.,1/4.,1/8.,  1 / 16.,1 / 8.,1 / 16. };
void SliceMagic::fuzzyMeans(Pixel* image, float* slice, int k) {
    Kcluster* clusters = kMeans(slice, k, 100);
    int kernel_size = 3;
    int ks2 = kernel_size * kernel_size;
    // First find each pixels belonging score to each cluster
    for (int i = 0; i < sizesq; i++) {
        image[i].fuzzy_cluster_scores = new float[k];
        for (int j = 0; j < k; j++) {
            image[i].fuzzy_cluster_scores[j] = clusters[j].belonging(image[i].getVal());
        }
    }
    float* new_vals = new float[sizesq];
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int* kernel_indexes = getKernelIndexes(x, y, kernel_size);
            float* scores = new float[k]();
            int index = xyToIndex(x, y);


            for (int i = 0; i < ks2; i++) {
                float dist = 1 + abs(image[index].getVal() - image[kernel_indexes[i]].getVal());

                for (int j = 0; j < k; j++) {
                    if (kernel_indexes[i] != -1) {
                        scores[j] += image[kernel_indexes[i]].fuzzy_cluster_scores[j] * gauss_kernel[i] *1 / (dist * dist );    // including dist2 removes ~300 of 9000 clusters
                        
                    }                        
                }
            }

            float best = 0;
            int best_index = 0;

            for (int i = 0; i < k; i++) {

                if (scores[i] > best) {
                    best = scores[i];
                    best_index = i;
                }
            }
            new_vals[index] = clusters[best_index].centroid;
            image[index].k_cluster = best_index;
            delete(kernel_indexes, scores);
        }
    }
    delete(clusters, new_vals);
}

float* SliceMagic::copySlice(float* slice) {
    float* copy = new float[size * size];
    for (int i = 0; i < sizesq; i++) {
        copy[i] = slice[i];
    }
    return copy;
}

float* SliceMagic::loadOriginal() {
    float* original = new float[load_size * load_size];
	Mat img = imread(im_path, cv::IMREAD_UNCHANGED);

    for (int y = 0; y < load_size; y++) {
        for (int x = 0; x < load_size; x++) {
            int hu = img.at<uint16_t>(y, x) - 32768;
            original[y*load_size+x] = hu;
        }
    }
    return original;
}


void SliceMagic::windowSlice(float* slice, float min, float max, int size) {
    for (int i = 0; i < size * size; i++) {
        float hu = slice[i];
        if (hu > max) { slice[i] = 1; }
        else if (hu < min) { slice[i] =0; }
        else { slice[i] = normVal(hu, min, max); }
    }
}

Color3* SliceMagic::colorConvert(float* slice, int size) {
    Color3* slice_ = new Color3[size];
    for (int i = 0; i < size; i++) {
        if (slice[i] <0 || slice[i] > 1)
        printf("%f\n", slice[i]);
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

void SliceMagic::showSlice(Color3* slice, string title, int s) {
    if (s == -1) s = size;
    Mat img(s, s, CV_8UC3);
    for (int y = 0; y < s; y++) {
        for (int x = 0; x < s; x++) {
            Color3 c = slice[xyToIndex(x, y, s)] * 255.;

                
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






void SliceMagic::histogramFuckingAround(float* slice) {
    Color3 viewColor = Color3(255, 105, 180)*(1./255.);
    float from;
    float to;
    Color3* image = new Color3[sizesq];
    while (true) {
        printf("From:\n");
        cin >> from;
        printf("To:\n");
        cin >> to;
        float* copy = copySlice(slice);
        for (int i = 0; i < sizesq; i++) {
            if (slice[i] >= from && slice[i] <= to)
                image[i] = viewColor;
            else
                image[i] = slice[i];
        }
        showSlice(image, "Fucking Around");
        waitKey();
    }
}




























void Pixel::checkAssignBelonging(Pixel* image) {
    if (!is_edge) return;
    int cluster_ = -1;
    Color3 color_;
    for (int i = 0; i < n_neighbors; i++) {
        Pixel neighbor = image[neighbor_indexes[i]];

        if (cluster_ == -1) {
            cluster_ = neighbor.cluster_id;
            color_ = neighbor.color;
        }
        else if (neighbor.cluster_id == cluster_ || neighbor.is_edge) {}
        else return;
    }
    if (cluster_ != -1) {// Else do nothing to this pixel
        assignToCluster(cluster_, color_);
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













void getMinMax(TissueCluster** clusters, int num_clusters, float* min, float* max) {
    *min = 99999999;
    *max = -9999999;
    for (int i = 0; i < num_clusters; i++) {
        float cmin = clusters[i]->getMin();
        float cmax = clusters[i]->getMax();
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

    delete(min, max);
    if (abs_dif < absolute_dif || rel_dif < relative_dif)
        return true;
    return false;
}

TissueCluster** mergeSublist(TissueCluster** main, int main_size, TissueCluster** addition, int addition_size) {
    TissueCluster** merged = new TissueCluster * [main_size + addition_size];
    for (int i = 0; i < main_size; i++) {
        merged[i] = main[i];
    }
    for (int i = 0; i < addition_size; i++) {
        merged[i + main_size] = addition[i];
    }
    // Too lazy to delete main and addition i guess...
    return merged;
}
int TissueCluster::mergeClusters(TissueCluster** clusters_sublist, Pixel* image, int num_clusters) {
    // Second redistribute each pixel to the survivor cluster
    if (isDeadmarked()) {
        printf("\n----------AM DEADMARKED, THIS SHOULD NOT HAPPEN------\n");
    }
    int merges = 0;
    for (int i = 0; i < num_clusters; i++) {
        cluster_id_is_neighbor[clusters_sublist[i]->cluster_id] = 0;
        if (clusters_sublist[i]->isDeadmarked() ) {
            printf("DEADMARKED, NO MERGE\n");
            continue;
        }
        if (clusters_sublist[i]->cluster_id == cluster_id) {
            printf("Same cluster\n");
            continue;
        }


        printf("\rExecuting merge of %05d into %05d", clusters_sublist[i]->cluster_id, cluster_id);
        for (int j = 0; j < clusters_sublist[i]->cluster_size; j++) {
            int pixel_index = clusters_sublist[i]->getPixel(j);
            addToCluster(image[pixel_index], image);
            image[pixel_index].assignToCluster(cluster_id, color);
            //image[pixel_index].color = Color3(255);
        }
        clusters_sublist[i]->deadmark(cluster_id);
        merges++;
    }
    recalcCluster(image);

    delete(clusters_sublist);   // Only deletes the sublist, original intact!
    return merges;
}

void copyPtr(int* from, int* to, int size) {
    for (int i = 0; i < size; i++) {
        to[i] = from[i];
    }
}
void copyPtr(float* from, float* to, int size) {
    for (int i = 0; i < size; i++) {
        to[i] = from[i];
    }
}
/*void TissueCluster::handleArraySize() {
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

        float* copyf = member_hu_values;
        member_values = new float[new_size];
        copyPtr(copyf, member_values, allocated_size);

        int* copy = member_pixel_indexes;
        pixel_indexes = new int[new_size];
        copyPtr(copy, pixel_indexes, allocated_size);

        allocated_size = new_size;
    }
}*/

void TissueCluster::addToCluster(Pixel pixel, Pixel* image) {
    //handleArraySize();
    //member_pixel_indexes[cluster_size] = pixel.index;
    //member_hu_values[cluster_size] = pixel.getVal();
    member_pixel_indexes.push_back(pixel.index);
    member_hu_values.push_back(pixel.getVal());

    int* pixel_neighbors = new int[9];
    int num_neighbors = pixel.getNeighbors(pixel_neighbors);
    for (int i = 0; i < num_neighbors; i++) {
        pixel_neighbors[i] = image[pixel_neighbors[i]].getID();
    }
    addPotentialNeighbors(pixel_neighbors, num_neighbors);
    cluster_size++;
    delete(pixel_neighbors);
}




TissueCluster** makeClusterSublist(TissueCluster* original, int* indexes, int size) {   // Verified it works
    TissueCluster** sub = new TissueCluster * [size];
    for (int i = 0; i < size; i++) {
        sub[i] = &original[indexes[i]];
    }
    return sub;
}
TissueCluster** TissueCluster::makeMergeableSublist(TissueCluster* clusters, int* mergeable_indexes, int num_mergeables) {  // Only necessary if the cluster is not merged immediately after, so change may have occured
    int* actual_indexes = new int[num_mergeables];
    int actual_num_mergeables = 0;
    for (int i = 0; i < num_mergeables; i++) { // Only find matches with a higher index!!!

        int actual_index = clusters[mergeable_indexes[i]].getSurvivingClusterID(clusters);
        if (actual_index != cluster_id) {
            actual_indexes[actual_num_mergeables] = actual_index;
            actual_num_mergeables++;
        }
    }
    num_mergeables = actual_num_mergeables;
    TissueCluster** sublist = makeClusterSublist(clusters, actual_indexes, actual_num_mergeables);
    delete(actual_indexes);
    return sublist;
}
TissueCluster** TissueCluster::findMergeables(TissueCluster* clusters, int num_clusters, float max_abs_dist, int* num_mergs) {
    //vector<int> merge_indexes;
    int* mergeable_indexes = new int[num_clusters];                    // BAAAAAAAAAAAAAAAAAAAAD
    int num_mergeables = 0;
    for (int i = 0; i < total_num_clusters; i++) { // Only find matches with a higher index!!!
        
        if (cluster_id_is_neighbor[i] && i != cluster_id) {
            if (clusters[i].isDeadmarked()) 
                continue;
                          
            float median_dif = abs(clusters[i].median - median);
            if (median_dif < max_abs_dist ) {
                mergeable_indexes[num_mergeables] = i;
                num_mergeables++;
            }
        }
    }
    //printf("%d mergeables found for cluster %d\n", num_mergeables, cluster_id);
    *num_mergs = num_mergeables;
    TissueCluster** sublist =  makeClusterSublist(clusters, mergeable_indexes, num_mergeables);
    delete(mergeable_indexes);


    return sublist;
}
