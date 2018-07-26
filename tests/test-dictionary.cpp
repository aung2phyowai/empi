/**********************************************************
 * University of Warsaw, Department of Biomedical Physics *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <cstdio>
#include <cstdlib>
#include "dictionary.hpp"

const double width = 2.0;
const double freqSampling = 128;
const int signalLength = 2048;
std::vector<double> envelope;

void assertEquals(double expected, double actual) {
	double diff = std::abs((actual - expected) / expected);
	if (diff > 1.0e-10) {
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
	const double scale = M_PI_4;
	const double amplitude0 = 1.1;
	const double amplitude1 = 0.01;
	const double phase = 0.2;
	for (int i=0; i<signalLength; ++i) {
		const double t = i/freqSampling - t0;
		const double t_s = t / scale;
		const double value = amplitude0 * sqrt(M_SQRT2/scale) * exp(-M_PI*t_s*t_s) * cos(2*M_PI*t*f0 + phase);
		channel.samples[i] = value;
	}
	double energy = computeEnergy(channel);
	for (int i=0; i<signalLength; ++i) {
		const double t = i/freqSampling - t1;
		const double t_s = t / scale;
		channel.samples[i] += amplitude1 * sqrt(M_SQRT2/scale) * exp(-M_PI*t_s*t_s) * cos(2*M_PI*t*f1);
	}

	MultiSignal signal;
	signal.channels.push_back(std::move(channel));

	EnvelopeGeneratorTemplate<EnvelopeGauss> generator;
	Dictionary dictionary(0.01, freqSampling, signalLength);
	DictionaryBlock block = dictionary.createBlock(&generator, scale, INFINITY);

/*
	workspace.compute(signal);
	Atoms atoms = workspace.findBestMatch(signal);
	Atom atom = atoms[0];

	assertEquals(energy, sqr(atom.params[0]));
	assertEquals(amplitude0, atom.params[1]);
	assertEquals(t0, atom.params[2] / freqSampling);
	assertEquals(scale, atom.params[3] / freqSampling);
	assertEquals(f0, 0.5 * atom.params[4] * freqSampling);
	assertEquals(phase, atom.params[5]);

	workspace.subtractAtom(atoms, signal);
	energy = computeEnergy(signal.channels[0]);

	atoms = workspace.findBestMatch(signal);
	atom = atoms[0];

	assertEquals(energy, sqr(atom.params[0]));
	assertEquals(amplitude1, atom.params[1]);
	assertEquals(t1, atom.params[2] / freqSampling);
	assertEquals(scale, atom.params[3] / freqSampling);
	assertEquals(f1, 0.5 * atom.params[4] * freqSampling);
 */
}
