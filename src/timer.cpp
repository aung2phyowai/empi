/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include "timer.hpp"

//----------------------------------------------------------------------

Timer::Timer(void)
{ }

void Timer::start(void) {
	start_ = clock();
}

double Timer::time(void) const {
	return time_;
}

void Timer::stop(void) {
	time_ += (clock() - start_) / static_cast<double>(CLOCKS_PER_SEC);
}

//----------------------------------------------------------------------

#ifndef NDEBUG
Timer timers[TIMER_COUNT];
#endif
