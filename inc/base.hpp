/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_BASE_HPP
#define	EMPI_BASE_HPP

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

//enum AtomType { ATOM_GABOR = 13 };
//
//struct Atom {
//	AtomType type;
//	std::vector<double> params;
//};

typedef double real;
typedef std::complex<real> complex;

class DictionaryBlock;

struct AtomParams {
		const DictionaryBlock* block;
		double scale;
		double frequency;
		double center;
};

/**
 * Fitted atom can be represented as:
 * g(n) = modulus · K · 1/√scale · f((n-n₀)/scale) · cos(…) = amplitude · f((n-n₀)/scale) · cos(…)
 * where f(n) is an envelope function defined as amplitude=1
 * while 1/√scale · f((n-n₀)/scale) is L²-normalized
 * and K · 1/√scale · f((n-n₀)/scale) · cos(…) is L²-normalized as well.
 */
struct AtomFit {
		double amplitude;
		double modulus;
		double phase;
};

struct Atom {
		AtomParams params;
		std::vector<AtomFit> fits;
};

struct AtomResult : public AtomParams, AtomFit {
    AtomResult(const AtomParams& params, const AtomFit& fit)
      : AtomParams(params), AtomFit(fit)
    { }
};

typedef std::vector<std::vector<AtomResult>> MultiChannelResult;

struct SingleSignal {
	double freqSampling;
	std::vector<double> samples;

	double computeEnergy() const {
		const int N = samples.size();
		double sum = 0.0;
		for (int i=0; i<N; ++i) {
			sum += samples[i] * samples[i];
		}
		return sum / freqSampling;
	}

	inline int size(void) const {
		return samples.size();
	}
	
	const double* data(void) const {
		return samples.data();
	}

	inline double operator[](int index) const {
		return samples[index];
	}
};

struct MultiSignal {
	std::vector<SingleSignal> channels;

	double computeEnergy() const {
		double sum = 0.0;
		for (const SingleSignal& channel : channels) {
			sum += channel.computeEnergy();
		}
		return sum;
	}

	double getFreqSampling() const {
		return channels.empty() ? NAN : channels[0].freqSampling;
	}

	int getSampleCount() const {
		return channels.empty() ? 0 : channels[0].samples.size();
	}
	
	inline const SingleSignal& operator[](int channel) const {
		return channels[channel];
	}
};

class Exception : public std::runtime_error {
public:
	explicit Exception(const std::string& camelCaseMessage)
	: std::runtime_error(camelCaseMessage) { }
};

#endif	/* EMPI_BASE_HPP */
