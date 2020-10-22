#include <conio.h>
#include <iostream>
#include <math.h>
using namespace std;

#define KEY_UP 72
#define KEY_DOWN 80
#define KEY_LEFT 75
#define KEY_RIGHT 77





class Camera {
public:
	float z = 0;
	float y = -1;
	float x = 0;

	float radius = 1;

	float pitch = 3.1415/2;
	float yaw = -3.1415 / 2;;
	double rotation_step = 2 * 3.1415 / 20; // 20 clicks per rotation

	void updatePos(char key_pressed);
private:
	void cylindricToCartesian();
};

class Environment {

public:
	Camera camera;

	void Run();
};