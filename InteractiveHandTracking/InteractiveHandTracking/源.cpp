#include"OpenGL_Display.h"
#include <tchar.h>
using namespace std;
using namespace chrono;
//共享内存的相关定义
HANDLE hMapFile;
LPCTSTR pBuf;
#define BUF_SIZE 1024
TCHAR szName[] = TEXT("Global\\MyFileMappingObject");    //指向同一块共享内存的名字
float *GetSharedMemeryPtr = nullptr;

void setSharedMemery();

void main(int argc, char** argv)
{
	setSharedMemery();
	GlobalSetting mGlobalSetting;
	mGlobalSetting.type = REALTIME;
	mGlobalSetting.start_points =3;
	mGlobalSetting.maxPixelNUM = 300;
	mGlobalSetting.sharedMeneryPtr = GetSharedMemeryPtr;
	mGlobalSetting.object_type = {redCube,yellowSphere,greenCylinder };


	TrackingManager* mTrackingManager = new TrackingManager(mGlobalSetting);

	DS::mTrackingManager = mTrackingManager;
	DS::init(argc, argv);
	DS::start();
}

void setSharedMemery()
{
#pragma region SharedMemery
	hMapFile = CreateFileMapping(
		INVALID_HANDLE_VALUE,    // use paging file
		NULL,                    // default security
		PAGE_READWRITE,          // read/write access
		0,                       // maximum object size (high-order DWORD)
		BUF_SIZE,                // maximum object size (low-order DWORD)
		szName);                 // name of mapping object

	if (hMapFile == NULL)
	{
		_tprintf(TEXT("Could not create file mapping object (%d).\n"),
			GetLastError());
		exit(0);
	}
	pBuf = (LPTSTR)MapViewOfFile(hMapFile,   // handle to map object
		FILE_MAP_ALL_ACCESS, // read/write permission
		0,
		0,
		BUF_SIZE);

	if (pBuf == NULL)
	{
		_tprintf(TEXT("Could not map view of file (%d).\n"),
			GetLastError());

		CloseHandle(hMapFile);
		exit(0);
	}

	GetSharedMemeryPtr = (float*)pBuf;
#pragma endregion SharedMemery
}