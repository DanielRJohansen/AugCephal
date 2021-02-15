#pragma once

#include "Constants.h"
#include "Containers.h"
#include <string>
#include <iostream>


//#define pi 3.1415

const int hu_indexes = HU_MAX - HU_MIN;


class ColorMaker
{
public:
	ColorMaker();
	inline Color colorFromHu(int hu) { return col_lut[hu - HU_MIN]; }
	inline int catFromHu(int hu) { return cat_lut[hu - HU_MIN]; }
	inline float belongingFromHu(int hu) { return belonging_lut[hu]; }
	Color forceColorFromCat(int cat, int prev_val);
private:
	Color colorFromValue();
	void initCategories();

	Color col_lut[hu_indexes];
	int cat_lut[hu_indexes];
	float belonging_lut[hu_indexes];
	void makeLUT();

	Category categories[NUM_CATS];
};

