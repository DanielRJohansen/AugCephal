#include "SliceMagic.h"
void onMouse(int event, int x, int y, int, void*);


float* global_hu_vals;

SliceMagic::SliceMagic() {
	original = new float[size*size];
    loadOriginal();

    int k = 10;
    int min_n = 2;

    global_hu_vals = copySlice(original);


    float* slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    showSlice(colorConvert(slice), "windowed");

    delete(slice);
    slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    rotatingMaskFilter(slice);
    showSlice(colorConvert(slice),  "RMFilter");

    delete(slice);
    slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    rotatingMaskFilter(slice);
    kMeans(slice, k);
    showSlice(colorConvert(slice), "RMFilter + " + to_string(k) + "-kmeans  ");
    waitKey();

    delete(slice);
    slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    medianFilter(slice);
    kMeans(slice, k);
    showSlice(colorConvert(slice), to_string(k)+"-kmeans");

    delete(slice);
    slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    kMeans(slice, k);
    requireMinNeighbors(slice, min_n);
    showSlice(colorConvert(slice), to_string(min_n)+"neighbors");


    delete(slice);
    slice = copySlice(original);
    windowSlice(slice, -500, 1000);
    kMeans(slice, k);
    requireMinNeighbors(slice, min_n*2);
    showSlice(colorConvert(slice), to_string(min_n*2) + "neighbors");

    waitKey();
}

void SliceMagic::rotatingMaskFilter(float* slice) {
    Mask masks[9];
    float* copy = copySlice(slice);
    printf("this far");
    int i = 0;
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            masks[i] = Mask(x, y);
            i++;
        }
    }

    for (int y = 2; y < size - 2; y++) {
        for (int x = 2; x < size - 2; x++) {
            if (slice[xyToIndex(x, y)] == 1 || slice[xyToIndex(x, y)] == 0)       // as to not erase bone or brigthen air
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
            float lowest_var = 999999;
            float kernel_copy[25];
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 25; j++)
                    kernel_copy[j] = kernel[j];
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

void SliceMagic::kMeans(float* slice, int k) {
    cluster* clusters = new cluster[k];
    for (int i = 0; i < k; i++) {
        clusters[i] = cluster((float)i/k);
    }

    //Iterate clustering
    for (int iter = 0; iter < 20; iter++) {
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
        else { slice[i] = (hu - min) / (max - min); }
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