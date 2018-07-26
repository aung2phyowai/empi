/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_WORKSPACE_HPP
#define	EMPI_WORKSPACE_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

#include "base.hpp"
#include "dictionary.hpp"
#include "envelope.hpp"
#include "fourier.hpp"
#include "heap.hpp"

//------------------------------------------------------------------------------
/*
class WorkspaceScale {
public:
		const double scale;
		const int Nfft;
		const std::vector<double> envelope;
		const int sampleStep;
		const long timeAtomCount;
		const int freqAtomCount;
		const size_t atomCount;

		WorkspaceScale(int Nfft, std::vector<double>&& envelope, std::vector<complex>&& envelopeFFT, int sampleStep, long timeAtomCount, int freqAtomCount, double scale) :
			scale(scale),
			Nfft(Nfft),
			envelope(envelope),
			sampleStep(sampleStep),
			timeAtomCount(timeAtomCount),
			freqAtomCount(freqAtomCount),
			atomCount(static_cast<size_t>(timeAtomCount) * static_cast<size_t>(freqAtomCount))
		{
			assert(Nfft > 0);
			assert(freqAtomCount <= Nfft/2 + 1);
			assert(envelope.size() <= static_cast<size_t>(Nfft));
		}
};

//------------------------------------------------------------------------------

class WorkspaceNormalization {
		const complex z;

public:
		WorkspaceNormalization(complex z) : z(z)
		{ }

		double value(double phase) const {
			return 0.5 * (1 + std::real(z * std::polar(1.0, 2*phase)));
		}
};

//------------------------------------------------------------------------------

class NormalizedScale {
		const double freqSampling;
		FourierFull* const fourierFull;
		FourierFreq* const fourierFreq;
		std::vector<complex> envelopeFFT;

public:
		const WorkspaceScale data;

		NormalizedScale(WorkspaceScale&& source, FourierFull* fourierFull, FourierFreq* fourierFreq, double freqSampling)
			: freqSampling(freqSampling), fourierFull(fourierFull), fourierFreq(fourierFreq), data(source)
		{
			FourierResultFull result = fourierFull->compute(data.Nfft, &data.envelope, data.envelope, false, 0, 1);
			envelopeFFT.insert(envelopeFFT.begin(), result.data[0], result.data[0]+result.K);
		}

		WorkspaceNormalization energy(int k) {
			const int K = static_cast<int>(envelopeFFT.size());
			assert(k >= 0);
			assert(k < K);
			int k2 = -k*2;
			if (k2 < -K) {
				k2 += data.Nfft;
			}
			int i0 = data.envelope.size()/2;
			complex z = (k2 >= 0) ? envelopeFFT[k2] : std::conj(envelopeFFT[-k2]);
			z *= std::polar(1.0, -4*M_PI*k*i0/data.Nfft) / freqSampling;
			return WorkspaceNormalization(z);
		}

		complex z(int k) const {
			const int K = static_cast<int>(envelopeFFT.size());
			assert(k >= 0);
			assert(k < K);
			int k2 = -k*2;
			if (k2 < -K) {
				k2 += data.Nfft;
			}
			complex z = (k2 >= 0) ? envelopeFFT[k2] : std::conj(envelopeFFT[-k2]);
			return z / freqSampling;
		}

		WorkspaceNormalization energy(double w) {
			int i0 = data.envelope.size()/2;
			complex z = fourierFreq->compute(&data.envelope, data.envelope, false, 0, -2.0*w, 1).data[0];
			z *= std::polar(1.0, -4*M_PI*w*i0) / freqSampling;
			return WorkspaceNormalization(z);
		}
};
*/

//------------------------------------------------------------------------------



class GSLVector : public std::shared_ptr<gsl_vector> {
public:
		explicit GSLVector(const size_t size)
			: std::shared_ptr<gsl_vector>(gsl_vector_alloc(size), gsl_vector_free)
		{ }
};

struct AtomParamsConverter {
    const DictionaryBlock* block;

    AtomParamsConverter(const DictionaryBlock* block)
      : block(block)
    { }

		GSLVector toGSL(const AtomParams& params) const {
      assert(params.block == block);
			GSLVector x(3);
			gsl_vector_set(x.get(), 0, std::log(params.scale) / std::log(block->scaleFactor));
			gsl_vector_set(x.get(), 1, params.frequency / block->stepInFreq);
			gsl_vector_set(x.get(), 2, params.center / block->stepInTime);
			return x;
		}

		AtomParams fromGSL(const gsl_vector* x) const {
      AtomParams params;
      params.block = block;
      params.scale = std::pow(block->scaleFactor, gsl_vector_get(x, 0));
			params.frequency = gsl_vector_get(x, 1) * block->stepInFreq;
			params.center = gsl_vector_get(x, 2) * block->stepInTime;
      return params;
		}
};

class Optimizer
{
public:
		Optimizer(const MultiSignal* signal,
              const AtomParamsConverter *converter, bool samePhase, double scaleMin, double scaleMax)
			: signal(signal), converter(converter), samePhase(samePhase), scaleMin(scaleMin), scaleMax(scaleMax), fourierFreq(signal->channels.size())
		{}

		static double fun(const gsl_vector *x, void *params)
		{
			Optimizer* self = reinterpret_cast<Optimizer*>(params);
			return -self->value(self->converter->fromGSL(x));
		}

		double value(const AtomParams &params, AtomFit *fit = nullptr)
		{
			if (params.frequency < 0 || params.scale < scaleMin || params.scale > scaleMax) {
				return 0;
			}
      params.block->generator->computeValues(params.scale, params.center, envelope);
			complex z = fourierFreq.computeNormFactor(envelope.values, params.frequency);
			FourierFrequencyView frequencyView = fourierFreq.compute(*signal, envelope.values, samePhase, envelope.offset, params.frequency);
			return matchFrequency(frequencyView, z, params.frequency, params.scale, envelope.shift, fit);
		}

		double optimize(AtomParams &params, AtomFit *fit = nullptr)
		{
//			printf("-- start optimize --\n");
			GSLVector x = converter->toGSL(params);
			std::shared_ptr<gsl_multimin_fminimizer> minimizer(
				gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2rand, 3),
				gsl_multimin_fminimizer_free
			);

			GSLVector step(3);
			gsl_vector_set(step.get(), 0, 0.1);
			gsl_vector_set(step.get(), 1, 0.1);
			gsl_vector_set(step.get(), 2, 0.1);

			gsl_multimin_function func;
			func.n = 3;
			func.f = Optimizer::fun;
			func.params = this;

			gsl_multimin_fminimizer_set(minimizer.get(), &func, x.get(), step.get());
			int iter = 0, status;
			do {
				if (++iter > 1000) {
					// maybe we have at least partial accuracy
					status = gsl_multimin_test_size(gsl_multimin_fminimizer_size(minimizer.get()), 1.0e-3);
					if (status == GSL_SUCCESS) {
						break;
					}
					// no, but iteration limit is exceeded
					throw Exception("minimizerTooManyIterations");
				}
				if (gsl_multimin_fminimizer_iterate(minimizer.get())) {
					// unexpected problem with iteration
					throw Exception("minimizerIterationError");
				}
				status = gsl_multimin_test_size(gsl_multimin_fminimizer_size(minimizer.get()), 1.0e-6);
			}
			while (status == GSL_CONTINUE);

			if (status != GSL_SUCCESS) {
				throw Exception("minimizerInvalidStatus");
			}

			params = converter->fromGSL(minimizer->x);
			return value(params, fit);
		}

private:
		const MultiSignal *signal;
		const AtomParamsConverter *converter;
		bool samePhase;
		double scaleMin, scaleMax;
		FourierFreq fourierFreq;

		Envelope envelope;
};


class Workspace {
		Dictionary* dictionary;
		FourierFull fourierFull;
		FourierFreq fourierFreq;
		const bool samePhase;
    const bool optimize;

		std::unique_ptr<Heap<double>> heap;

public:
    std::pair<int, int> subtractAtomFromSignal(const AtomParams& params, const AtomFit& fit, SingleSignal& signal);

//    double optimize(const MultiSignal& signal, AtomParams& params, AtomFit* fit = nullptr);

		Workspace(Dictionary* dictionary, int channelCount, bool samePhase, bool optimize) :
      dictionary(dictionary),
			fourierFull(channelCount, dictionary->blocks.back().samplesForFFT, FFTW_MEASURE),
			fourierFreq(channelCount),
			samePhase(samePhase),
      optimize(optimize)
		{ }

    AtomParams getAtomsByIndex(size_t heapIndex);

		void compute(const MultiSignal& signal);

    Atom findBestMatch(const MultiSignal& signal);

		void subtractAtom(const Atom& atom, MultiSignal& signa);

    size_t getAtomCount() {
      return dictionary->getAtomCount();
    }
};

//------------------------------------------------------------------------------

/*
class WorkspaceBuilder {
		const double energyError;
		const double scaleMin;
		const double scaleMax;
		const double freqMax;

public:
		WorkspaceBuilder(double energyError, double scaleMin, double scaleMax, double freqMax)
			: energyError(energyError), scaleMin(scaleMin), scaleMax(scaleMax), freqMax(freqMax) { }

		Workspace* prepareWorkspace(std::shared_ptr<EnvelopeGenerator> generator, double freqSampling, int channelCount, int sampleCount, bool samePhase) const;
};
*/

//------------------------------------------------------------------------------

#endif /* EMPI_WORKSPACE_HPP */
