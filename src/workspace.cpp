/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <algorithm>
#include <list>
#include <memory>

#include "workspace.hpp"
#include "timer.hpp"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

//------------------------------------------------------------------------------

void Workspace::compute(const MultiSignal& signal) {
	std::vector<double> values;

	size_t heapIndex = 0;
	for (auto& block : dictionary->blocks) {
		values.resize(heapIndex + block.atomsAll);
		const int sampleShift = lrint(block.envelope.shift);
		int sampleCenter = 0;
		for (int it=0; it<block.atomsInTime; ++it, sampleCenter+=block.stepInTime) {
			FourierSpectrumView spectrumView = fourierFull.compute(block.samplesForFFT, signal, block.envelope.values, samePhase, sampleCenter-sampleShift);
			const int K = block.atomsInFreq;
			for (int k=0; k<K; ++k) {
				double square = block.matchFrequencyBin(spectrumView[k], k);
				values[heapIndex++] = square;
			}
		}
	}

	heap.reset( new Heap<double>(std::move(values)) );

}

//struct Atom {
//    double amplitude;
//    double energy;
//    long sampleCenter;
//    double sampleScale;
//
//};

AtomParams Workspace::getAtomsByIndex(size_t heapIndex) {
  for (const auto& block : dictionary->blocks) {
    if (heapIndex < block.atomsAll) {
      const int k = heapIndex % block.atomsInFreq;
      const int it = heapIndex / block.atomsInFreq;

      AtomParams params;
      params.block = &block;
      params.scale = block.scale;
      params.frequency = static_cast<double>(k) / static_cast<double>(block.samplesForFFT);
      params.center = it * block.stepInTime;
      return params;
    }
    heapIndex -= block.atomsAll;
  }
  throw Exception("atomOutsideWorkspace");
}

Atom Workspace::findBestMatch(const MultiSignal& signal) {
	if (!heap.get()) {
		throw Exception("internalInitializationError");
	}
  AtomParams paramsBest;
  double energyError = dictionary->energyError;
  double resultBest = 0;

  if (optimize) {
    heap->walk([this, energyError, &paramsBest, &resultBest, &signal](HeapItem<double> item, double& min) {
        AtomParams params = getAtomsByIndex(item.key);
        AtomParamsConverter converter(params.block);
        Optimizer optimizer(&signal, &converter, samePhase, params.block->scaleMin, params.block->scaleMax);
        double result = optimizer.optimize(params);
        if (result > resultBest) {
          resultBest = result;
          paramsBest = params;
          min = (1 - energyError) * resultBest;
          // TODO sprawdzić czy da się sensownie określić minimalną skalę s w próbkach
        }
    });
  } else {
    HeapItem<double> item = heap->peek();
    paramsBest = getAtomsByIndex(item.key);
    resultBest = item.value;

//    int offset = lrint(paramsBest.sCenter * dictionary->hzSampling) - paramsBest.block->envelope.size()/2;
//    double w = paramsBest.hzFrequency / dictionary->hzSampling;
//    FourierFrequencyView frequencyView = fourierFreq.compute(signal, paramsBest.block->envelope, samePhase, offset, w);
//    printf("1 fourier[0] = %lf\n", std::abs(frequencyView[0]));
//    paramsBest.block->matchFrequencyValue(frequencyView, w);
  }

  if (resultBest <= 0) {
    throw Exception("signalIsEmpty"); // TODO czy na pewno wyjątek?
  }

  std::vector<AtomFit> fits(signal.channels.size());
  AtomParamsConverter converter(paramsBest.block);
  Optimizer optimizer(&signal, &converter, samePhase, paramsBest.block->scaleMin, paramsBest.block->scaleMax);
  optimizer.value(paramsBest, fits.data()); // to powinno być równe resultBest
  return Atom{paramsBest, fits};
}

std::pair<int, int> Workspace::subtractAtomFromSignal(const AtomParams& params, const AtomFit& fit, SingleSignal& signal) {
	Envelope envelope;
	std::vector<double>& waveform = envelope.values;
  params.block->generator->computeValues(params.scale, params.center, envelope);
	const int Nw = waveform.size();
	const long sampleOffset = envelope.offset;
  const double w = params.frequency;
  const double c = fit.amplitude * sqrt(params.scale);
	for (int i=0; i<Nw; ++i) {
		waveform[i] *= c * std::cos(2*M_PI*w*(i-envelope.shift) + fit.phase);
	}

	const long iL = std::max(0L, -sampleOffset);
	const long iR = std::min(
		static_cast<long>(Nw),
		static_cast<long>(signal.samples.size())-sampleOffset
	);
  // TODO long/int
	for (int i=iL; i<iR; ++i) {
		signal.samples[sampleOffset+i] -= waveform[i];
	}
	return std::make_pair(sampleOffset+iL, sampleOffset+iR-1);
}

void Workspace::subtractAtom(const Atom& atom, MultiSignal& signal) {
	TIMER_START(subtractAtomFromSignal);
	const int channelCount = signal.channels.size();
	std::pair<int, int> iLR;
	for (int c=0; c<channelCount; ++c) {
		// each channel should return the same index
		iLR = subtractAtomFromSignal(atom.params, atom.fits[c], signal.channels[c]);
	}
	TIMER_STOP(subtractAtomFromSignal);

	std::list<std::pair<size_t, double>> valuesToUpdate;

	size_t heapIndexScale = 0;
	for (const auto& block : dictionary->blocks) {
		int Nw = block.envelope.values.size();
		int sampleShift = Nw/2; // TODO może tu też trzeba brać z block.envelope ?

		long itL = std::max(0, (iLR.first - sampleShift + block.stepInTime-1) / block.stepInTime);
		long itR = std::min(block.atomsInTime-1, (iLR.second + sampleShift) / block.stepInTime);
		long sampleCenter = itL * block.stepInTime;
		size_t heapIndex = heapIndexScale + itL * block.atomsInFreq;
		for (int it=itL; it<=itR; ++it, sampleCenter+=block.stepInTime) {
			TIMER_START(subtractAtom_FFT);
			FourierSpectrumView spectrumView = fourierFull.compute(block.samplesForFFT, signal, block.envelope.values, samePhase, sampleCenter-sampleShift);
			TIMER_STOP(subtractAtom_FFT);
			TIMER_START(subtractAtom_COMPUTE);
			const int K = block.atomsInFreq;
			for (int k=0; k<K; ++k) {
        double square = block.matchFrequencyBin(spectrumView[k], k);
				valuesToUpdate.push_back(std::make_pair(heapIndex++, square));
			}
			TIMER_STOP(subtractAtom_COMPUTE);
		}
		heapIndexScale += block.atomsAll;
	}
	TIMER_START(subtractAtom_UPDATE);
	for (const auto& pair : valuesToUpdate) {
		heap->update(pair.first, pair.second);
	}
	TIMER_STOP(subtractAtom_UPDATE);
}

//------------------------------------------------------------------------------

/**
 * Minimum scale parameter for Gabor atoms, in number of samples.
 */
/*
static const double MIN_SCALE_IN_SAMPLES = 3.0;

Workspace* WorkspaceBuilder::prepareWorkspace(std::shared_ptr<EnvelopeGenerator> generator, double freqSampling, int channelCount, int sampleCount, bool samePhase) const {
	double scaleMin = std::max(this->scaleMin, MIN_SCALE_IN_SAMPLES / freqSampling);
	double scaleMax = std::min(this->scaleMax, sampleCount / freqSampling);
	double root = std::sqrt(-2.0/M_PI * std::log(1.0-energyError));
	double a = generator->computeScaleFactor(1.0 - energyError);

	std::vector<WorkspaceScale> scales;
	for (double s=scaleMin; s<=scaleMax; s*=a) {
		const double dt = root * s;
		const double df = root / s;
		long Nfft = fftwRound( std::lrint(freqSampling/df + 0.5) );
		if (Nfft > static_cast<long>(std::numeric_limits<int>::max())) {
			throw Exception("dictionaryIsTooFineForThisDecomposition");
		}
		std::vector<double> envelopeValues;
		generator->computeValues(s, 1.0/freqSampling, envelopeValues);
		long envelopeSize = static_cast<long>(envelopeValues.size());
		while (Nfft < envelopeSize) Nfft *= 2;

		const long sampleStep = lround(dt*freqSampling - 0.5);
		assert(sampleStep > 0);
		int timeAtomCount = (sampleCount + sampleStep - 1) / sampleStep;
		long freqAtomCount = Nfft / 2 + 1;
		if (std::isfinite(freqMax) && freqMax > 0) {
			freqAtomCount = std::min(freqAtomCount, std::lrint(freqMax/df - 0.5) + 1);
		}

		scales.emplace_back(
			Nfft,
			std::move(envelopeValues),
			sampleStep,
			timeAtomCount,
			freqAtomCount,
			s
		);
	}
	return new Workspace(channelCount, freqSampling, samePhase, scales, generator);
}
*/
