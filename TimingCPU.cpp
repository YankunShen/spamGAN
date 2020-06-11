/**************/
/* TIMING CPU */
/**************/

#include "TimingCPU.h"

#include <sys/time.h>
#include <iostream>

TimingCPU::TimingCPU(): cur_time_(0) { StartCounter(); }


void TimingCPU::StartCounter()
{
    struct timeval time;
    if(gettimeofday( &time, 0 )) return;
    cur_time_ = 1000000 * time.tv_sec + time.tv_usec;
}

double TimingCPU::GetCounter()
{
    struct timeval time;
    if(gettimeofday( &time, 0 )) return -1;

    long cur_time = 1000000 * time.tv_sec + time.tv_usec;
    double sec = (cur_time - cur_time_) / 1000000.0;
    if(sec < 0) sec += 86400;
    cur_time_ = cur_time;

    return 1000.*sec;
}

