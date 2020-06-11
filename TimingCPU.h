// 1 micro-second accuracy
// Returns the time in seconds

// #ifndef __TIMINGCPU_H__
// #define __TIMINGCPU_H__


class TimingCPU {

    private:
        long cur_time_;

    public:

        TimingCPU();

        ~TimingCPU();

        void StartCounter();

        double GetCounter();
};

