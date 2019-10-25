#include"InputManager.h"
#include"OpenGL_Display.h"

void main(int argc, char** argv)
{
	InputManager* mInputManager = new InputManager(REALTIME);

	DS::mInputManager = mInputManager;
	DS::init(argc, argv);
	DS::start();
}