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