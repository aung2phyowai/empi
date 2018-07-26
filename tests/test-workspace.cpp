/**********************************************************
 * University of Warsaw, Department of Biomedical Physics *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "workspace.hpp"

const double width = 2.0;
const double freqSampling = 128;
const int signalLength = 2048;
std::vector<double> envelope;

void assertEquals(double expected, double actual) {
	double diff = fabs(actual - expected);
	if (diff > 1.0e-7) {
		printf("DIFF! %lf instead of %lf\n", actual, expected);
		exit(1);
	}
}

inline double sqr(double x) {
	return x * x;
}

double computeEnergy(const SingleSignal& channel) {
	double energy = 0.0;
	for (int i=0; i<signalLength; ++i) {
		double value = channel.samples[i];
		energy += value * value;
	}
	return energy;
}

int main(void) {
	SingleSignal channel;
	channel.freqSampling = freqSampling;
	channel.samples.resize(signalLength);

	const double t0 = 6.0;
	const double t1 = 10.0;
	const double f0 = 4.0;
	const double f1 = 64.0;
	const double sScale = M_PI_4;
	const double amplitude0 = 1.1;
	const double amplitude1 = 0.01;
	const double phase = 0.2;
	for (int i=0; i<signalLength; ++i) {
		const double t = i/freqSampling - t0;
		const double t_s = t / sScale;
		const double value = amplitude0 * exp(-0.5*M_PI*t_s*t_s) * cos(2*M_PI*t*f0 + phase);
		channel.samples[i] = value;
	}
	double energy = computeEnergy(channel);
	for (int i=0; i<signalLength; ++i) {
		const double t = i/freqSampling - t1;
		const double t_s = t / sScale;
		channel.samples[i] += amplitude1 * exp(-0.5*M_PI*t_s*t_s) * cos(2*M_PI*t*f1);
	}
  double sampleScale = sScale * freqSampling;

	MultiSignal signal;
	signal.channels.push_back(std::move(channel));

  EnvelopeGeneratorTemplate<EnvelopeGauss> generator;
  Dictionary dictionary(0.01, signalLength);
  dictionary.addBlocks(&generator, sampleScale, sampleScale);

  Workspace workspace(&dictionary, 1, false, false);
	workspace.compute(signal);

	Atom atom = workspace.findBestMatch(signal);
	assertEquals(energy, sqr(atom.fits[0].modulus));
	assertEquals(amplitude0, atom.fits[0].amplitude);
	assertEquals(t0, atom.params.center / freqSampling);
	assertEquals(sampleScale, atom.params.scale);
	assertEquals(f0, atom.params.frequency * freqSampling);
	assertEquals(phase, atom.fits[0].phase);
	workspace.subtractAtom(atom, signal);
	energy = computeEnergy(signal.channels[0]);

	atom = workspace.findBestMatch(signal);
	assertEquals(energy, sqr(atom.fits[0].modulus));
  assertEquals(amplitude1, atom.fits[0].amplitude);
  assertEquals(t1, atom.params.center / freqSampling);
  assertEquals(sampleScale, atom.params.scale);
  assertEquals(f1, atom.params.frequency * freqSampling);
  assertEquals(0.0, atom.fits[0].phase);

  workspace.subtractAtom(atom, signal);
  energy = computeEnergy(signal.channels[0]);
  assertEquals(0.0, energy);
}
