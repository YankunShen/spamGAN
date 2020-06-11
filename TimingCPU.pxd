
cdef extern from "TimingCPU.h":
    cdef cppclass TimingCPU:
        TimingCPU();
        long cur_time_;
        void StartCounter();
        double GetCounter();