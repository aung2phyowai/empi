/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_TIMER_HPP
#define EMPI_TIMER_HPP

#include <ctime>

class Timer {
	clock_t start_;
	double time_;

public:
	Timer(void);

	void start(void);

	void stop(void);

	double time(void) const;
};

#ifdef NDEBUG
#define TIMER_START(NAME)
#define TIMER_STOP(NAME)
#define PRINT_TIMER(NAME)
#else

#define TIMER_COUNT 8
#define TIMER_subtractAtom_FFT 0
#define TIMER_subtractAtom_COMPUTE 1
#define TIMER_subtractAtom_UPDATE 2
#define TIMER_subtractAtomFromSignal 3
#define TIMER_subtractAtom 4
#define TIMER_findBestMatch 5
#define TIMER_compute 6
#define TIMER_prepareWorkspace 7
#define TIMER_subtractAtom_OTHER 8

extern Timer timers[TIMER_COUNT];

#define TIMER_START(NAME) timers[TIMER_ ## NAME].start()
#define TIMER_STOP(NAME) timers[TIMER_ ## NAME].stop()
#define PRINT_TIMER(NAME) std::cerr << #NAME << '\t' << timers[TIMER_ ## NAME].time() << std::endl

#endif  /* NDEBUG */

#endif  /* EMPI_TIMER_HPP */
