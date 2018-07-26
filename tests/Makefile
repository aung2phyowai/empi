CXXFLAGS=--std=c++11 -Wall -Wextra -I../inc -O2
LDLIBS=-lfftw3 -lgsl -lgslcblas
.PHONY : default

default : test-fourier test-gauss test-heap test-norm test-workspace

test-gauss : test-gauss.cpp ../src/gabor.cpp

test-norm : test-norm.cpp ../src/gabor.cpp

test-workspace : test-workspace.cpp ../src/timer.cpp ../src/workspace.cpp

test-workspace2 : test-workspace2.cpp ../src/timer.cpp ../src/workspace.cpp
