#include"OpenGL_Display.h"

void main(int argc, char** argv)
{
	GlobalSetting mGlobalSetting;
	mGlobalSetting.type = REALTIME;
	mGlobalSetting.start_points = 3;
	mGlobalSetting.maxPixelNUM = 192;
	mGlobalSetting.object_type = redCube;


	TrackingManager* mTrackingManager = new TrackingManager(mGlobalSetting);

	DS::mTrackingManager = mTrackingManager;
	DS::init(argc, argv);
	DS::start();
}