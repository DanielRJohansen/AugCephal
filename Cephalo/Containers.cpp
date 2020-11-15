#include "Containers.h"




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



Category::Category(int idd, float sta, float sto, Color c) {
	id = idd;
	start = sta; 
	stop = sto;
	color = c;
	centroid = (start + stop) / 2;
	//variance = (stop - start) / 2 * var_scalar;
}


ColorScheme::ColorScheme() {
	Category lung(0, -700, -120, Color(212, 163, 163)); //Shoudl be 45
	Category fat(1, -120, -90, Color(225, 203, 52));
	Category fluids(2, -90, 20, Color(128, 128, 200));
	Category muscle(3, 20, 55, Color(201, 99, 99)); //Should be 35
	Category clot(4, 55, 200, Color(161, 44, 44));
	Category bone(5, 200, 2000, Color(255, 255, 255));
	Category cats[6] = {lung, fat, fluids, muscle, clot, bone };
	categories = new Category[6];

	for (int i = 0; i < 6; i++) {
		categories[i] = cats[i];
	}

	float hu;
	for (int i = 0; i < 700; i++) {
		colors[i] = Color(100, 100, 100);
		hu = (i + lower_limit);
		for (int j = 0; j < 6; j++) {
			
			if (hu > categories[j].start && hu<= categories[j].stop) {
				float diff = ((float)i - categories[j].centroid)/(700) * categories[j].var_scalar;;
				colors[i] = categories[j].color;// .mul(1 + diff);
				cat_indexes[i] = categories[j].id;
				break;
			}
		}
	}
}