/**********************************************************
 * University of Warsaw, Department of Biomedical Physics *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <cstdio>
#include <cstdlib>
#include "fourier.hpp"
#include "gabor.hpp"

const double width = 1.0;
const double freqSampling = 512;

int main(void)
{
	std::vector<double> envelope = make_gauss_envelope(width, 1.0/freqSampling);
	// we assume the envelope to be normalized
	int length = static_cast<int>(envelope.size());

	int Nfft = 1 << 12;
	assert(length <= Nfft);

	FourierFull fourierFull(1, Nfft, FFTW_ESTIMATE);
	FourierFreq fourierFreq(1, length);
	FourierResultFull result = fourierFull.compute(Nfft, &envelope, envelope, false, 0);

	NormalizedScale scale(
		WorkspaceScale(Nfft, std::move(envelope), 1, 1, Nfft/2+1, width),
		&fourierFull,
		&fourierFreq,
		freqSampling
	);

	for (int k=0; k<result.K; ++k) {
		double w = static_cast<double>(k) / static_cast<double>(Nfft);
		WorkspaceNormalization norm1 = scale.energy(k);
		WorkspaceNormalization norm2 = scale.energy(w);
		for (double phase=0; phase<2*M_PI; phase+=0.6283) {
			int i0 = length/2;
			double energyCalculated = 0.0;
			for (int i=0; i<length; ++i) {
				double value = envelope[i] * cos(2*M_PI*k*(i-i0)/Nfft + phase);
				energyCalculated += value * value;
			}
			energyCalculated /= freqSampling;
			double energyEstimated1 = norm1.value(phase);
			double energyEstimated2 = norm2.value(phase);
			double diff = std::max(
				energyEstimated1 - energyCalculated,
				energyEstimated2 - energyCalculated
			);
			if (diff > 1.0e-10) {
				printf("ERROR: diff = %le for k=%d phase=%lf\n", diff, k, phase);
				exit(1);
			}
			if (!k) break; // only zero phase is valid at zero frequency
		}
	}
}
