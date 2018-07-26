/**********************************************************
 * University of Warsaw, Department of Biomedical Physics *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <cstdio>
#include "fftw.hpp"

const int INPUT = 1024;
const int OUTPUT = INPUT / 2 + 1;

int main(void) {
	fftwDouble input(INPUT);
	fftwComplex output(OUTPUT);
	fftwPlan plan(INPUT, input, output, FFTW_ESTIMATE);

	for (int i=0; i<INPUT; ++i) {
		input[i] = rand() / (double) RAND_MAX;
	}
	plan.execute();
	for (int k=0; k<OUTPUT; ++k) {
		std::complex<double> single = dftSingleValue(input, k);
		double diff = std::abs(output[k] - single);
		if (diff > 1.0e-10) {
			printf("ERROR: diff = %lf\n", diff);
			return 1;
		}
	}
}
