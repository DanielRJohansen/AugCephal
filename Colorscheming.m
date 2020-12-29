%% COLOR FUNCTION


clc
lung = [-700, -600];
fat = [-120, -90];
fluids = [-30 15];
water = [-2, 2 ];
muscle = [30 55];
blood = [13 50 ];
hematoma = [50 100];
clot = [50 75 ];
cancellous = [300 400];
cortical = [1000 1900];
foreign = [2500 3000];
noise = [-700 1500];
cols = ['r' 'y' 'c' 'b' 'r' 'r' 'r' 'k' 'k' 'g'];
cats = [lung; fat;fluids; water; muscle; blood; hematoma; clot; cancellous; cortical; foreign; ];
catnames = ["lung", "fat", "fluids", "water" "muscle" "blood" "hematoma" "clot" "cancellous" "cortical" "foreign"];
min_belonging = 0.0001;

x = -1000:0.2:2000;
for i = 1:length(cats)
   spread = cats(i,2)-cats(i,1);
   center = (cats(i,1) + cats(i,2))/2;
   sd = spread/4;
   f = 1/(sd*sqrt(2*3.1415)) *exp(-((x-center).^2)/(2*sd^2));
   f = plot(x, f);
   %legend(catnames(i))
   hold on
end
%set(gca, 'YScale', 'log')
ylabel("Inclusion")
xlabel("Hounsfield unit")
legend(catnames)
axis([-100 100 min_belonging 0.08])
hold off

%%  ACTIVATION FUNCTION
clc
blocks = 9*5;
x = 1:1:blocks+3;
f = 2./(1+exp(-x/15))-1;
plot(x, f)



%% ACCUMULATED ALPHA FUNCTION
x = 0:1:10;
y = 1./x.^2;
plot(x,y)







%% Image normalization
y = [0.000000, 388, 320, 279, 231, 204, 188, 168, 165, 136, 143, 109, 114, 103, 106, 80, 91, 87, 84, 77, 66, 66, 64, 64, 54, 43, 52, 35, 27, 35, 28, 29, 32, 37, 25, 20, 16, 21, 8, 12, 10, 14, 31, 14, 14, 19, 19, 14, 16, 17, 16, 16, 17, 6, 11, 9, 18, 21, 16, 26, 15, 21, 19, 9, 17, 16, 20, 19, 14, 19, 20, 18, 19, 19, 15, 23, 13, 16, 13, 17, 31, 16, 21, 21, 16, 28, 18, 18, 20, 27, 17, 27, 26, 22, 23, 21, 23, 27, 22, 27, 18, 25, 19, 30, 23, 27, 25, 32, 27, 27, 22, 14, 27, 16, 41, 25, 23, 37, 30, 14, 30, 30, 16, 34, 31, 29, 30, 25, 26, 23, 28, 28, 26, 28, 32, 35, 32, 25, 24, 38, 17, 34, 40, 28, 39, 38, 31, 32, 28, 38, 35, 27, 40, 28, 39, 42, 29, 54, 35, 53, 27, 56, 45, 36, 60, 49, 85, 68, 56, 114, 90, 128, 150, 148, 182, 161, 215, 219, 333, 279, 422, 555, 699, 695, 718, 873, 1024, 1250, 1383, 1562, 1824, 1968, 2238, 2429, 2643, 2741, 2901, 3045, 3179, 3532, 3702, 3896, 3643, 3813, 3842, 4117, 3981, 3988, 3919, 3834, 3611, 3539, 3428, 3381, 3017, 2829, 2565, 2386, 2279, 2035, 2005, 1924, 1682, 1491, 1468, 1265, 1210, 1044, 961, 965, 927, 966, 895, 925, 871, 945, 938, 808, 866, 913, 863, 900, 960, 937, 957, 1008, 1068, 1081, 1120, 1042, 1314, 1357, 1371, 1439, 1603, 1514, 1760, 1769, 2123, 1995, 2202, 2340, 2659, 2724, 3079, 3207, 3418, 3710, 4253, 4554, 4692, 5153, 5319, 5502, 6281, 6455, 6603, 6561, 6952, 6864, 6830, 7501, 7205, 7330, 7266, 7006, 6796, 6462, 6357, 6003, 5975, 5575, 5508, 5268, 4744, 4376, 3871, 3655, 3360, 2836, 2583, 2316, 1974, 1695, 1578, 1310, 1227, 901, 910, 739, 680, 552, 463, 388, 394, 286, 241, 194, 168, 158, 92, 128, 92, 78, 77, 102, 72, 67, 53, 52, 50, 52, 59, 29, 30, 40, 48, 33, 45, 42, 24, 36, 33, 33, 26, 37, 32, 39, 31, 31, 34, 33, 30, 42, 30, 23, 28, 30, 27, 26, 36, 29, 25, 38, 33, 36, 36, 41, 35, 36, 35, 47, 33, 67, 56, 47, 67, 60, 47, 42, 76, 66, 98, 97, 126, 124, 101, 106, 121, 110, 139, 115, 132, 102, 118, 105, 103, 142, 147, 156, 147, 130, 105, 105, 150, 117, 92, 86, 93, 76, 106, 74, 64, 41, 57, 55, 53, 50, 51, 59, 68, 71, 68, 88, 101, 79, 59, 90, 96, 68, 91, 93, 102, 99, 91, 102, 124, 103, 123, 138, 132, 150, 108, 126, 98, 76, 74, 98, 104, 101, 66, 73, 67, 80, 55, 52, 75, 42, 11, 11, 4, 7, 0, 7, 0, 1, 0, 0, 1, 9, 0, 0, 0, 1, 0, 1, 0, 2, 1, 7, 4, 2, 7, 6, 13, 10, 12, 16, 12, 14, 34, 24, 22, 32, 31, 40, 32, 34, 24, 31];
binsize = 0.002;
ylog = log(y);
x = binsize:binsize:1;
bar(x, ylog)



