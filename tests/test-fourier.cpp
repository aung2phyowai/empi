/**********************************************************
 * University of Warsaw, Department of Biomedical Physics *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <cstdio>
#include "fourier.hpp"

const int CHANNELS = 16;
const int INPUT = 1024;
const bool SAME_PHASE = false;

int main(void) {
	std::vector<std::vector<double>> inputs;
	for (int c=0; c<CHANNELS; ++c) {
		inputs.emplace_back(INPUT);
	}

	FourierFull fourier_full(CHANNELS, INPUT, FFTW_ESTIMATE);
	FourierFreq fourier_freq(CHANNELS);
	for (int c=0; c<CHANNELS; ++c) {
		for (int i=0; i<INPUT; ++i) {
			inputs[c][i] = rand() / (double) RAND_MAX - 0.5;
		}
	}

	std::vector<double> envelope(INPUT-1, 1.0);
	FourierSpectrumView full = fourier_full.compute(INPUT, inputs, envelope, INPUT/2, SAME_PHASE);
	for (int k=0; k<full.K; ++k) {
		double w = (double) k / (double) INPUT;
		FourierFrequencyView freq = fourier_freq.compute(inputs, envelope, INPUT/2, SAME_PHASE, w);
		for (int c=0; c<CHANNELS; ++c) {
			double diff = std::abs(full[k][c] - freq[c]);
			if (diff > 1.0e-10) {
				printf("ERROR: diff = %lf\n", diff);
				return 1;
			}
		}
	}
}
