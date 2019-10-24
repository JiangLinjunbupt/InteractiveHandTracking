#include"RealSenseSR300.h"

void main()
{
	Camera* mCamera = new Camera(REALTIME);
	RealSenseSensor* mRealSenseSensor = new RealSenseSensor(mCamera);

	mRealSenseSensor->start();

	long long i = 0;
	while (true)
	{
		i++;
	}
	cout << i << endl;
}