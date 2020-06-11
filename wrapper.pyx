# distutils: language = c++
# distutils: sources = TimingCPU.cpp

from TimingCPU cimport TimingCPU

cdef class PyTimingCPU:
    cdef TimingCPU *c_time

    def  __cinit__(self):
        self.c_time = new TimingCPU()
    
    def StartCounter(self):
        return self.c_time.StartCounter()
    
    def GetCounter(self):
        return self.c_time.GetCounter()
    
    @property
    def cur_time_(self):
        return self.c_rect.cur_time_
    @cur_time_.setter
    def cur_time_(self, cur_time_):
        self.c_rect.cur_time_ = cur_time_


