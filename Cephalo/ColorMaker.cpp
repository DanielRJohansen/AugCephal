#include "ColorMaker.h"



ColorMaker::ColorMaker() {
	initCategories();
	makeLUT();
}


void ColorMaker::initCategories() {
	Category lung(0, 0, -700, -600, Color(0xf2, 0xd7, 0xd0));

	Category fat(1, 1, -120, -90, Color(241, 194, 125));

	Category fluids(2, 2, -30, 15, Color(116, 204, 244));
	Category water(3, 2, -2, 2, Color(35, 137, 218));

	Category muscle(4, 3, 25, 55, Color(0x8a, 0x03, 0x03)); // Priority, as smaller pieces will be reassigned during clustering!
	Category bloodclot(5, 3, 50, 75, Color(0x47, 0x11, 0x04));
	Category hematoma(6, 3, 50, 100, Color(41, 9, 55));		// Priority from rarity, as
	Category blood(7, 3, 13, 50, Color(0x5c, 0x10, 0x04));

	Category cancellous(8, 4, 300, 400, Color(222, 202, 176));

	Category cortical(9, 5, 1000, 1900, Color(255, 255, 255));

	//Category noisy(10, -700, 1500, Color(0, 255, 0));
	Category foreign(10, 6, 2500, 3000, Color(249, 166, 2));

	// LOADING ORDER DETERMINES CAT PRIORITY!!!!!!!!!!!!!!!!!!!!!!!!!!
	Category cats[NUM_CATS] = { lung, fat, fluids, water, muscle, bloodclot,
		hematoma, blood, cancellous, cortical, foreign };

	for (int i = 0; i < NUM_CATS; i++) {
		categories[i] = cats[i];
	}
}

void ColorMaker::makeLUT() {
	for (int i = 0; i < hu_indexes; i++) {
		int hu_val = i + HU_MIN;
		Color c(0, 0, 0);
		float total_belonging = 0;
		float highest_belonging = 0;

		for (int cat_i = 0; cat_i < NUM_CATS; cat_i++) {
			Category cat = categories[cat_i];
			float sd = cat.spread/4.;
			float x = hu_val;
			float exponential = - (x - cat.center) * (x - cat.center) / (2 * sd * sd);
			//printf("%f         %f\n", exponential, expf(exponential));
			float belonging_score = 1 / (sd * sqrt(2 * pi)) * expf(exponential);

			if (belonging_score > highest_belonging) {
				cat_lut[i] = cat_i;
				highest_belonging = belonging_score;
			}
			total_belonging += belonging_score;
			c = c + categories[cat_i].color * belonging_score;
		}
		c = c * (1 / total_belonging);
		col_lut[i] = c;


		
		if (total_belonging < 0.00000001) { cat_lut[i] = UNKNOWN_CAT; }
		//cout << total_belonging << endl;
		float blutscore = total_belonging * BELONGING_COEFFICIENT;
		//if (hu_val < 200)
		//	printf("%d     %f\n", hu_val, blutscore);
		if (blutscore > 1) { blutscore = 1; }
		belonging_lut[i] = blutscore;
	}
}

Color ColorMaker::forceColorFromCat(int cati, int prev_val) {
	Category cat = categories[cati];
	int center = (cat.start + cat.stop) / 2;
	if (prev_val > center) { return col_lut[cat.stop]; }
	return col_lut[cat.start];
}