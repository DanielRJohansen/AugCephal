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
    applyCanny(slice);
    showSlice(colorConvert(slice), "windowed");

    slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    deNoiser(slice);
    rotatingMaskFilter(slice, 14);
    applyCanny(slice);
    showSlice(colorConvert(slice), "denoised");
    waitKey();

    slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    rotatingMaskFilter(slice, 14);
    showSlice(colorConvert(slice), "RMFiltermany");

    slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    rotatingMaskFilter(slice,14);
    applyCanny(slice);
    showSlice(colorConvert(slice),  "Canny");



    waitKey();


}

float cm1[25] = { 1, 0.2, 0, 0, 0,  0.2, 1, 0.2, 0, 0,  0, 0.2, 1, 0.2, 0,  0, 0, 0.2, 1, 0.2,  0, 0, 0, 0.2, 1 };
float cm2[25] = { 0, 0, 0, 0.2, 1,  0, 0, 0.2, 1, 0.2,  0, 0.2, 1, 0.2, 0,  0.2, 1, 0.2, 0, 0,  1, 0.2, 0, 0, 0 };
float cm3[25] = { 0, 0, 0, 0, 0,  0.2, 0.2, 0.2, 0.2, 0.2,  1, 1, 1, 1, 1,  0.2, 0.2, 0.2, 0.2, 0.2,  0, 0, 0, 0, 0 };
float cm4[25] = { 0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0,  0, 0.2, 1, 0.2, 0 };
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
                    //cout << " " << i;
                }
            }
            //cout << endl;
            slice[xyToIndex(x, y)] = best_mean;
        }
    }

}

inline int radToIndex(float rad) {
    return round((rad + 3.1415) / (2 * 3.1415) * 8);
}
inline bool isLegal(int x, int y) { return x * y > 0 && x < 512 && y < 512; }

void SliceMagic::nonMSStep(vector<int>* edge_indexes, float* steepest_edge, int* steepest_index, float* G, float* theta, bool* fb, int inc, int x, int y, int x_step, int y_step, int step_index){
    for (int i = inc; i < size; i += inc) {
        int x_ = x + x_step * i;
        int y_ = y + y_step * i;
        int pp_index = xyToIndex(x_, y_);
        if (!isLegal(x_, y_)) { return; }

        float grad = abs(G[pp_index]);
        int step_index_ = radToIndex(theta[xyToIndex(x_, y_)]);
        if (grad > grad_threshold) {
        //if (grad > min_val && step_index == step_index_) {
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

void SliceMagic::nonMS(float* slice, float* G, float* theta, bool* forced_black) {
    int x_steps[9] = { 1, 1, 0, -1, -1, -1, 0, 1, 1 };
    int y_steps[9] = { 0, 1, 1, 1, 0, -1, -1, -1, 0 };

    vector<int>* edge_indexes = new vector<int>;
    float* steepest_edge = new float;
    int* steepest_index = new int;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int p_index = xyToIndex(x, y);
            if (G[p_index] < grad_threshold) { continue; }

            int step_index = radToIndex(theta[xyToIndex(x, y)]);
            int x_step = x_steps[step_index];
            int y_step = y_steps[step_index];


            edge_indexes->push_back(p_index); 
            *steepest_edge = G[p_index];
            *steepest_index = p_index;  

            nonMSStep(edge_indexes, steepest_edge, steepest_index, G, theta, forced_black, 1, x, y, x_step, y_step, step_index);
            nonMSStep(edge_indexes, steepest_edge, steepest_index, G, theta, forced_black, -1, x, y, x_step, y_step, step_index);

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

struct int2 {
    int2() {}
    int2(int x, int y) : x(x), y(y) {}
    int x, y;
};
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
    //nonMS(slice, G, theta, forced_black);
    hysteresis(slice, G, forced_black);
    nonMS(slice, G, theta, forced_black);

}


//              USE FOR COLERING
/*
void SliceMagic::propagate(float* slice, bool* forced_black, int* cat, int x, int y, int id) {
    cat[xyToIndex(x, y)] = id;

    for (int y_ = y - 1; y_ <= y + 1; y_++) {
        for (int x_ = x - 1; x_ <= x + 1; x_++) {
            if (!isLegal(x_, y_))
                continue;
            if (cat[xyToIndex(x_, y_)] == -1 && !forced_black[xyToIndex(x, y)]) { //unassigned
                propagate(slice, forced_black, cat, x_, y_, id)
            }
        }
    }
}

void SliceMagic::hysteresis(float* slice, bool* forced_black) {
    int* cat = new int[sizesq];
    for (int i = 0; i < sizesq; i++) { cat[i] = -1; }

    int id = 0;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int index = xyToIndex(x, y);

            if (cat[index == -1] && slice[index] == 0.) {
                propagate(slice, forced_black, cat, x, y, id);
            }
            id++;
        }
    }
    propagate(slice, forced_black, cat, x, y);
}*/


void SliceMagic::kMeans(float* slice, int k, int iterations) {
    cluster* clusters = new cluster[k];
    for (int i = 0; i < k; i++) {
        clusters[i] = cluster((float)i/k);
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
    //if (event != CV_EVENT_LBUTTONDOWN)
      //  return;

    Point pt = Point(x, y);
    std::cout << "(" << pt.x << ", " << pt.y << ")      huval: "<< global_hu_vals[y*512+x] << '\n';

}