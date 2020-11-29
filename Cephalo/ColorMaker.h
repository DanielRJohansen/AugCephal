#pragma once

#include "Constants.h"
#include "Containers.h"
#include <string>
#include <iostream>

#define e 2.71828
#define pi 3.1415

const int NUM_CATS = 12;



class ColorMaker
{
public:
	ColorMaker();
	inline Color colorFromHu(int hu) {
		//cout << hu << "     ";
		return hu_lut[hu - min_hu]; }
	inline int catFromHu(int hu) { return cat_lut[hu - min_hu]; }
private:
	Color colorFromValue();
	void initCategories();

	int min_hu = -700;
	Color hu_lut[10700];
	int cat_lut[10700];
	void makeLUT();

	Category categories[NUM_CATS];
};

