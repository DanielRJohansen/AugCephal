%x = -700:2000;


clc
lung = [-700, -600];
fat = [-120, -90];
fluids = [-30 15];
water = [-2, 2 ];
blood = [13 50 ];
hematoma = [35 100];
clot = [50 75 ];
cancellous = [300 400];
cortical = [1500 1900];
foreign = [2000 10000];
noise = [-700 1500];
cols = ['r' 'y' 'c' 'b' 'r' 'r' 'r' 'k' 'k' 'g'];
cats = [lung; fat;fluids; water; blood; hematoma; clot; cancellous; cortical; foreign; ];


x = -1000:0.2:2000
for i = 1:length(cats)
   spread = cats(i,2)-cats(i,1);
   center = (cats(i,1) + cats(i,2))/2;
   %sd = sqrt(spread);
   sd = spread/4
   %fplot(@(x) i*x*0.001)
   %fplot(@(x) log(1/(sd*sqrt(2*3.1415)) *exp(-((x-center)^2)/(2*sd^2))), [-1000 2000])
   %fp = fplot(@(x) 1/(sd*sqrt(2*3.1415)) *exp(-((x-center)^2)/(2*sd^2)), [-1000 500]);
   f = 1/(sd*sqrt(2*3.1415)) *exp(-((x-center).^2)/(2*sd^2));
   f = plot(x, f)
   %fp.Color = cols(i);
   hold on
end
axis([-200 500 0 0.1])
%axis([-400 100 0 0.15])
hold off

%flung = @(x) 1/()
%%
%fplot(@(x) 1/(25*sqrt(2*3.1415)) *exp(-((x+650)^2)/(2*25^2)), [-1000 2000])
1/(0.8*sqrt(2*3.1415)) * exp(-((0)^2)/(2*0.8^2))
%axis([-1000 1000 0 0.04])