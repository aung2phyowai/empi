/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_ENVELOPE_HPP
#define	EMPI_ENVELOPE_HPP

#include <limits>
#include "base.hpp"

//------------------------------------------------------------------------------

/**
 * Gaussian envelope function f(x) = exp(−½πx²)
 */
class EnvelopeGauss {
public:
	/**
	 * Compute a single value f(x) of the envelope function.
	 * @param x
	 * @return f(x)
	 */
	static double computeValue(double x) {
		return (x>=-3 && x<=+3) ? exp(-0.5*M_PI*x*x) : 0.0;
	}

	/**
	 * Solve equation ∫ f(x√a) f(x/√a) dx = √(2a/(a²+1)) = I.
	 * @param I
	 * @return scale factor a
	 */
	static double computeScaleFactor(double I) {
		const double I2 = I * I;
		return (1 + sqrt(1 - I2*I2)) / I2;
	}
};

//------------------------------------------------------------------------------

struct Envelope {
		// offset between signal start and envelope start
		int offset;

		// offset between envelope start and center
		double shift;

		// values
		std::vector<double> values;
};

//------------------------------------------------------------------------------

class EnvelopeGenerator {
public:
    /**
     * Max envelope length (in samples) to make sure
     * that even after rounding up to the power of 2
     * the sample count will still fit in int.
     */
    static const int MAX_LENGTH = std::numeric_limits<int>::max()/2 + 1;

	/**
	 * Compute discrete values of the scaled envelope function
	 * i.e. 1/√s f((i-i0)/s)
	 *
	 * @param width  scale parameter in samples
	 * @param envelope  buffer for the resulting values
	 */
	virtual void computeValues(double scale, double center, Envelope& envelope) =0;

	/**
	 * Solve equation ∫ f(x√a) f(x/√a) dx = I,
	 * where f(x) is the envelope function.
	 * @param I
	 * @return scale factor a
	 */
	virtual double computeScaleFactor(double I) const =0;
};

//------------------------------------------------------------------------------

template<class ENVELOPE>
class EnvelopeGeneratorTemplate : public EnvelopeGenerator {
	std::vector<double> pos, neg;

public:
	/**
	 * Compute discrete values of the scaled envelope function
	 * i.e. 1/√s f((i-i0)/s)
	 * and i0 corresponds to the middle of the resulting values (index i/2).
	 * This function will always generate odd number of values.
	 * Resulting envelope is guaranteed to be not longer than 2^30 samples.
	 *
	 * @param width  scale parameter in samples
	 * @param envelope  buffer for the resulting values
	 */
	void computeValues(double scale, double center, Envelope& envelope) override
	{
		int indexCenter = (center > 0) ? lround(center - 0.5) : lround(center + 0.5) - 1;
		double fractionalShift = center - indexCenter;

		neg.clear();
		pos.clear();
		double value;
		for (int n=0; (value = ENVELOPE::computeValue((n-fractionalShift)/scale)) > 0; --n) {
			neg.push_back(value);
		}
		for (int n=1; (value = ENVELOPE::computeValue((n-fractionalShift)/scale)) > 0; ++n) {
			pos.push_back(value);
		}
		size_t size = neg.size() + pos.size();
		if (size > MAX_LENGTH) {
			throw Exception("maximal scale is too large");
		}
		int sizeNeg = neg.size();
		const double norm = 1.0 / sqrt(scale);
		envelope.offset = indexCenter - sizeNeg + 1;
		envelope.shift = sizeNeg - 1 + fractionalShift;
		envelope.values.clear();
		envelope.values.reserve(size);
		for (auto i=neg.rbegin(); i!=neg.rend(); ++i) {
			envelope.values.push_back(*i * norm);
		}
		for (auto i=pos.begin(); i!=pos.end(); ++i) {
			envelope.values.push_back(*i * norm);
		}
	}

	/**
	 * Solve equation ∫ f(x√a) f(x/√a) dx = I,
	 * where f(x) is the envelope function.
	 * @param I
	 * @return scale factor a
	 */
	double computeScaleFactor(double I) const override {
		return ENVELOPE::computeScaleFactor(I);
	}
};

#endif	/* EMPI_ENVELOPE_HPP */
