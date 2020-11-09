#include "Containers.h"


void Cluster::mergeCluster(Cluster c2) {
	//Assign parent cluster too all blocks contained in c2
	//Delete c2
	//Do NOT recalc center, this would be cheating for sime pixels.
}

Color Color::cutOff(Color c) {
	if (c.r > 255) c.r = 255;
	if (c.g > 255) c.g = 255;
	if (c.b > 255) c.b = 255; 
	return c;
}
Color Color::add(Color a) {
	Color c = Color(r + a.r, g + a.g, b + a.b);
	return cutOff(c);
}
Color Color::mul(float s) {
	return cutOff(Color(r * s, g * s, b * s));
}



Category::Category(string n, float sta, float sto, Color c) {
	name = n;
	start = sta; 
	stop = sto;
	color = c;
	centroid = (start + stop) / 2;
	variance = (stop - start) / 2 * var_scalar;
}


ColorScheme::ColorScheme() {
	Category lung("lung", -700, -120, Color(212, 163, 163)); //Shoudl be 45
	Category fat("fat", -120, -90, Color(225, 203, 52));
	Category fluids("fluids", -90, 20, Color(128, 128, 200));
	Category muscle("muscle", 20, 55, Color(201, 99, 99)); //Should be 35
	Category clot("clot", 55, 200, Color(161, 44, 44));
	Category bone("bone", 200, 2000, Color(255, 255, 255));
	Category categories[6] = {lung, fat, fluids, muscle, clot, bone };
	//Category categories[2] = { fat, muscle };
	float hu;
	for (int i = 0; i < 700; i++) {
		colors[i] = Color(100, 100, 100);
		hu = (i + lower_limit);
		for (int j = 0; j < 6; j++) {
			
			if (hu > categories[j].start && hu< categories[j].stop) {
				//float diff = ((float)i - categories[j].centroid) * categories[j].variance;;
				colors[i] = categories[j].color;// .mul(1 + diff);
				//names[i] = categories[j].name;
				break;
			}
		}
		cout << colors[i].r << endl;
	}
}