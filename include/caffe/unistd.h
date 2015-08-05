/** This file is part of the Mingw32 package.
 *  unistd.h maps     (roughly) to io.h
 */
#ifndef _UNISTD_H
#define _UNISTD_H
#include <io.h>
#include <process.h>
#define NOMINMAX
#include <windows.h>
#define usleep(x) Sleep(x/1000.0)
/*
int usleep(unsigned long usec)
{
	Sleep(usec / 1000.0);
}*/

#endif /* _UNISTD_H */