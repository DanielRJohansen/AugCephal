#include "ColorMaker.h"



ColorMaker::ColorMaker() {
	initCategories();
	makeLUT();
}


void ColorMaker::initCategories() {
	Category lung(0, -700, -600, Color(0xf2, 0xd7, 0xd0)); //Shoudl be 45
	Category fat(1, -120, -90, Color(241, 194, 125));
	Category fluids(2, -30, 15, Color(116, 204, 244));
	Category water(3, -2, 2, Color(35, 137, 218));
	Category hematoma(4, 35, 100, Color(0x32, 0x0f, 0x03));
	Category bloodclot(5, 50, 75, Color(0x47, 0x11, 0x04));
	Category blood(6, 13, 50, Color(0x5c, 0x10, 0x04));
	Category muscle(7, 35, 55, Color(0x8a, 0x03, 0x03)); //Should be 35
	Category cancellous(8, 300, 400, Color(222, 202, 176));
	Category cortical(9, 1800, 1900, Color(255, 255, 255));
	Category foreign(10, 500, 30000, Color(249, 166, 2));

	Category cats[NUM_CATS] = { lung, fat, fluids, water, hematoma, bloodclot,
	blood, muscle, cancellous, cortical, foreign };

	for (int i = 0; i < NUM_CATS; i++) {
		categories[i] = cats[i];
	}
}

void ColorMaker::makeLUT() {
	for (int i = 0; i < 30700; i++) {
		int hu_val = i - min_hu;
		Color c(0, 0, 0);
		float total_belonging = 0;
		float highest_belonging = 0;

		for (int cat_i = 0; cat_i < NUM_CATS; cat_i++) {
			Category cat = categories[cat_i];
			float center = cat.start;
			int spread = cat.stop - cat.start;
			float sd = spread / 4.;
			float x = hu_val;
			float belonging_score = 2.5 / (sd * sqrt(2 * pi)) * powf(e, 
				-powf((x - center), 2) / 
				(2 * powf(spread, 2)));

			if (belonging_score > highest_belonging) {
				cat_lut[i] = cat_i;
				highest_belonging = belonging_score;
			}
			total_belonging += belonging_score;
			c = c + categories[cat_i].color * belonging_score;
		}
		c = c * (1 / total_belonging);
		hu_lut[i] = c;
	}
	printf("Colormaker initiated");
}
