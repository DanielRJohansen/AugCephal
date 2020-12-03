%% COLOR FUNCTION


clc
lung = [-700, -600];
fat = [-120, -90];
fluids = [-30 15];
water = [-2, 2 ];
blood = [13 50 ];
hematoma = [50 100];
clot = [50 75 ];
cancellous = [300 400];
cortical = [1000 1900];
foreign = [2500 3000];
noise = [-700 1500];
cols = ['r' 'y' 'c' 'b' 'r' 'r' 'r' 'k' 'k' 'g'];
cats = [lung; fat;fluids; water; blood; hematoma; clot; cancellous; cortical; foreign; ];


x = -1000:0.2:2000
for i = 1:length(cats)
   spread = cats(i,2)-cats(i,1);
   center = (cats(i,1) + cats(i,2))/2;
   sd = spread/4
   f = 1/(sd*sqrt(2*3.1415)) *exp(-((x-center).^2)/(2*sd^2));
   f = plot(x, f);
   hold on
end
axis([-200 500 0 0.1])
hold off

%%  ACTIVATION FUNCTION
clc
blocks = 9*5;
x = 1:1:blocks+3;
f = 2./(1+exp(-x/15))-1;
plot(x, f)
