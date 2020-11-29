#include "ColorMaker.h"



ColorMaker::ColorMaker() {
	initCategories();
	makeLUT();
}


void ColorMaker::initCategories() {
	Category lung(0, -620, -600, Color(0xf2, 0xd7, 0xd0)); //Shoudl be 45
	Category fat(1, -120, -90, Color(241, 194, 125));
	Category fluids(2, -30, 15, Color(116, 204, 244));
	Category water(3, -2, 2, Color(35, 137, 218));
	Category blood(4, 13, 50, Color(0x5c, 0x10, 0x04));
	Category muscle(5, 35, 55, Color(0x8a, 0x03, 0x03)); //Should be 35
	Category hematoma(6, 35, 100, Color(0x32, 0x0f, 0x03));
	Category bloodclot(7, 50, 75, Color(0x47, 0x11, 0x04));
	Category cancellous(8, 300, 400, Color(222, 202, 176));
	Category cortical(9, 1500, 1900, Color(255, 255, 255));

	Category noisy(10, -700, 1500, Color(0, 255, 0));
	Category foreign(11, 2000, 10000, Color(249, 166, 2));

	Category cats[NUM_CATS] = { lung, fat, fluids, water, hematoma, bloodclot,
	blood, muscle, cancellous, cortical, noisy, foreign };

	for (int i = 0; i < NUM_CATS; i++) {
		categories[i] = cats[i];
	}
}

void ColorMaker::makeLUT() {
	for (int i = 0; i < 10700; i++) {
		int hu_val = i + min_hu;
		Color c(0, 0, 0);
		float total_belonging = 0;
		float highest_belonging = 0;

		for (int cat_i = 0; cat_i < NUM_CATS; cat_i++) {
			Category cat = categories[cat_i];
			float center = (cat.start + cat.stop)/2;
			float spread = cat.stop - cat.start;
			//float sd = sqrt(spread);
			float sd = spread;
			//cout << sd << endl;
			float x = hu_val;
			float belonging_score = 10 / (sd * sqrt(2 * pi)) * powf(e, 
				-powf((x - center), 2) / 
				(2 * powf(sd, 2.)));

			if (belonging_score > highest_belonging) {
				cat_lut[i] = cat_i;
				highest_belonging = belonging_score;
			}
			total_belonging += belonging_score;
			c = c + categories[cat_i].color * belonging_score;
		}
		//cout << total_belonging << endl;
		c = c * (1 / total_belonging);
		if (i < 2000) {
			printf("HUval: %d      %f       %f         %f     belongs to: %d.    Fraction: %f\n",
				hu_val, c.r, c.g, c.b, categories[cat_lut[i]].id, highest_belonging/total_belonging);
		}

		hu_lut[i] = c;
	}
	printf("Colormaker initiated");
}
