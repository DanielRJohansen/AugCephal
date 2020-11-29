#pragma once

#include "Constants.h"
#include "Containers.h"
#include <string>
#include <iostream>

#define e 2.71828
#define pi 3.1415

const int NUM_CATS = 11;



class ColorMaker
{
public:
	ColorMaker();
	inline Color colorFromHu(int hu) {
		if (hu < -700 || hu >= 30000) cout << "Wierd hu: "<< hu << endl;
		return hu_lut[hu + min_hu]; }
	inline int catFromHu(int hu) { return cat_lut[hu + min_hu]; }
private:
	Color colorFromValue();
	void initCategories();

	int min_hu = -700;
	Color hu_lut[30700];
	int cat_lut[30700];
	void makeLUT();

	Category categories[NUM_CATS];
};

