#include "mvt.h"

#include <sys/time.h>

unsigned long g_tic[NUM_OF_TIMER];
unsigned long g_toc[NUM_OF_TIMER];

unsigned long g_tic_count[NUM_OF_TIMER];
unsigned long g_toc_count[NUM_OF_TIMER];

#define TVAL(x) (x.tv_sec*1000000 + x.tv_usec)

void tic_all()
{
	timeval tval;
	gettimeofday(&tval, NULL);
	for( int i=0; i<NUM_OF_TIMER; i++)
	{
		g_tic[i] = TVAL(tval);
	}
}

unsigned long tic(ENUM_TIMER timer)
{
	timeval tval;
	gettimeofday(&tval, NULL);
	g_tic[timer] = TVAL(tval);
	return g_tic[timer];
}

unsigned long toc(ENUM_TIMER timer)
{
	timeval tval;
	gettimeofday(&tval, NULL);

	g_toc[timer] = (TVAL(tval)-g_tic[timer]);
	return g_toc[timer];
}

void tic_count(ENUM_TIMER timer)
{
	timeval tval;
	gettimeofday(&tval, NULL);

	g_tic_count[timer] = TVAL(tval);
}

void toc_count(ENUM_TIMER timer)
{
	timeval tval;
	gettimeofday(&tval, NULL);

	g_toc_count[timer] += (TVAL(tval)-g_tic_count[timer]);
}

void count_init(ENUM_TIMER timer)
{
	g_toc_count[timer] = 0;
}

unsigned long count(ENUM_TIMER timer)
{
	return g_toc_count[timer];
}

