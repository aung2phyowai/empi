/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/

/*
GaborWorkspaceMap::GaborWorkspaceMap(double s, int Nfft, int fCount, int tCount, double freqSampling, double tMax, int channelCount, MultichannelConstraint constraint)
:	TimeFreqMap<complex>(channelCount, fCount, tCount),
	Nfft(Nfft), input(Nfft), output(Nfft/2+1),
	plan(Nfft, input, output, FFTW_ESTIMATE | FFTW_DESTROY_INPUT),
	s(s), freqSampling(freqSampling), atomCount(fCount * tCount), constraint(constraint)
{
	for (int ti=0; ti<tCount; ++ti) {
		t(ti) = ti * tMax / (tCount - 1);
	}
	for (int fi=0; fi<fCount; ++fi) {
		f(fi) = fi * freqSampling / Nfft;
	}
}

void GaborWorkspaceMap::compute(const MultiSignal& signal) {
	for (int tIndex=0; tIndex<tCount; ++tIndex) {
		for (int cIndex=0; cIndex<cCount; ++cIndex) {
			compute(signal.channels[cIndex], cIndex, tIndex);
		}
	}
	index.reset( new GaborWorkspaceIndex(this) );
}

void GaborWorkspaceMap::compute(const SingleSignal& signal, int cIndex, int tIndex) {
	const long N = signal.samples.size();
	const double t0 = t(tIndex);
	const double hwGabor = GAUSS_HALF_WIDTH * s;
	// type casts are safe since we know signal length fits in int
	int iL = static_cast<int>( std::max(0L, std::lrint((t0 - hwGabor) * signal.freqSampling)) );
	int iR = static_cast<int>( std::min(N-1, std::lrint((t0 + hwGabor) * signal.freqSampling)) );
	const double t0fixed = t0 - iL / signal.freqSampling;

	input.zero();
	for (int i=iL; i<=iR; ++i) {
		input[i-iL] = signal.samples[i];
	}
	gaussize(input.data(), iR-iL+1, 1.0/signal.freqSampling, t0fixed, s);
	plan.execute();

	const double norm = 1.0 / signal.freqSampling;
	const double mult = 2 * M_PI * t0fixed;
	for (int fIndex=0; fIndex<fCount; ++fIndex) {
		value(cIndex, fIndex, tIndex) = norm * output[fIndex] * std::polar(1.0, f(fIndex) * mult);
	}
	if (index) {
		index->mark(tIndex);
	}
}

complex GaborWorkspaceMap::compute(double s0, double f0, double t0, const SingleSignal& signal) const {
	const long N = signal.samples.size();
	const double hwGabor = GAUSS_HALF_WIDTH * s0;
	// type casts are safe since we know signal length fits in int
	int iL = static_cast<int>( std::max(0L, std::lrint((t0 - hwGabor) * signal.freqSampling)) );
	int iR = static_cast<int>( std::min(N-1, std::lrint((t0 + hwGabor) * signal.freqSampling)) );
	const double t0fixed = t0 - iL / signal.freqSampling;

	input.zero();
	for (int i=iL; i<=iR; ++i) {
		input[i-iL] = signal.samples[i];
	}
	gaussize(input.data(), iR-iL+1, 1.0/signal.freqSampling, t0fixed, s0);

	const double norm = 1.0 / signal.freqSampling;
	const double mult = 2 * M_PI * t0fixed;
	double fIndex = input.size() * f0 / signal.freqSampling;
	complex out = dftSingleValue(input, fIndex);
	return norm * out * std::polar(1.0, f0 * mult);
}

TimeFreqValue GaborWorkspaceMap::max(const MultiSignal& signal) {
	if (!index) {
		throw Exception("internalLogicError");
	}
	return index->max(signal);
}
*/

//------------------------------------------------------------------------------

/*
void GaborWorkspaceIndex::update(void) {
	for (int fIndex=0; fIndex<map->fCount; ++fIndex) {
		GaborComputer updater(map, fIndex);
		for (int tIndex=0; tIndex<map->tCount; ++tIndex) if (tIndexNeedUpdate[tIndex]) {
			size_t key = fIndex * static_cast<size_t>(map->tCount) + tIndex;
			double value = updater.compute(tIndex, buffer);
			heap->update(key, value);
		}
	}
	std::fill(tIndexNeedUpdate.begin(), tIndexNeedUpdate.end(), false);
	tIndexNeedUpdateFlag = false;
}

GaborWorkspaceIndex::GaborWorkspaceIndex(const GaborWorkspaceMap* map)
: map(map), buffer(map->cCount) {
	tIndexNeedUpdate.resize(map->tCount);
	std::vector<double> data(map->tCount * map->fCount);
	size_t key = 0;
	for (int fIndex=0; fIndex<map->fCount; ++fIndex) {
		GaborComputer updater(map, fIndex);
		for (int tIndex=0; tIndex<map->tCount; ++tIndex) {
			data[key++] = updater.compute(tIndex, buffer);
		}
	}
	this->heap.reset(new Heap<double>(std::move(data)));
}

void GaborWorkspaceIndex::mark(int tIndex) {
	tIndexNeedUpdate[tIndex] = true;
	tIndexNeedUpdateFlag = true;
}

#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>

struct minParams {
	const GaborWorkspaceMap* map;
	const MultiSignal* signal;
	std::vector<complex>* buffer;
};

struct minResult {
	double s, f, t;
	double value;
};

double minFunc(const gsl_vector* x, void* params) {
	double s = x->data[0];
	double f = x->data[1];
	double t = x->data[2];
	minParams* p = static_cast<minParams*>(params);
	const GaborWorkspaceMap* map = p->map;
	const MultiSignal* signal = p->signal;
	if (f < 0 || s <= 0 || t < 0 || t >= signal->getSampleCount() / signal->getFreqSampling()) return 0.0;
	std::vector<complex>& buffer = *p->buffer;

	for (int c=0; c<map->cCount; ++c) {
		buffer[c] = map->compute(s, f, t, signal->channels[c]);
	}
	double freqNyquist = 0.5 * map->freqSampling;
	bool isHighFreq = f > 0.5 * freqNyquist;
	double normComplex = std::sqrt(M_SQRT2 / s);
	double f4Norm = isHighFreq ? (freqNyquist - f) : f;
	double expFactor = std::exp(-2*M_PI*s*s*f4Norm*f4Norm);
	if (map->constraint && !isHighFreq) {
		(*map->constraint)(buffer);
	}
	double square = 0.0;
	for (int cIndex=0; cIndex<map->cCount; ++cIndex) {
		double phase = std::arg(buffer[cIndex]);
		double phase4Norm = isHighFreq ? (2*M_PI*freqNyquist*t - phase) : phase;
		double normReal = M_SQRT2 * normComplex / std::sqrt(1.0 + std::cos(2*phase4Norm) * expFactor);
		double moduli = std::abs(buffer[cIndex]) * normReal * std::sqrt(map->freqSampling);
		square += moduli * moduli;
	}
	return -square;
}

TimeFreqValue GaborWorkspaceIndex::max(const MultiSignal& signal) {
	if (tIndexNeedUpdateFlag) {
		printf("UPDATING!\n");
		update();
	}
	HeapItem<double> peak;
	printf("s = %lf\n", map->s);
	
	gsl_vector* v = gsl_vector_alloc(3);
	gsl_vector* step = gsl_vector_alloc(3);
	// TODO accurate step size
	step->data[0] = 0.1 * map->s;
	step->data[1] = map->f(1) - map->f(0);
	step->data[2] = map->t(1) - map->t(0);
	minParams params{map, &signal, &buffer};
	gsl_multimin_fminimizer* minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex, 3);
	gsl_multimin_function func;
	func.f = minFunc;
	func.n = 3;
	func.params = &params;
	std::vector<complex> buffer(map->cCount);
	heap->walk([this, &signal, &peak, &buffer, &func, v, step, minimizer](HeapItem<double> item, double& min) {
		v->data[0] = map->s;
		v->data[1] = map->f(item.key / map->tCount);
		v->data[2] = map->t(item.key % map->tCount);

		gsl_multimin_fminimizer_set(minimizer, &func, v, step);
		printf("%lf %lf %lf  %lf...\n",
				gsl_multimin_fminimizer_x(minimizer)->data[0],
				gsl_multimin_fminimizer_x(minimizer)->data[1],
				gsl_multimin_fminimizer_x(minimizer)->data[2],
				item.value);
		for (int it=0; it<100 && gsl_multimin_fminimizer_iterate(minimizer) != GSL_ENOPROG; ++it) {
			// TODO break if we have move far away (to the next cell and beyond)
			// or return 0.0 from minFunc
		}
		printf("... %lf  %lf %lf %lf\n",
				gsl_multimin_fminimizer_minimum(minimizer),
				gsl_multimin_fminimizer_x(minimizer)->data[0],
				gsl_multimin_fminimizer_x(minimizer)->data[1],
				gsl_multimin_fminimizer_x(minimizer)->data[2]);
		printf("END MINIMIZATION\n");

		peak = item;
		min = std::max(min, 0.8 * item.value); // TODO replace 0.8 with precise value
	});
	gsl_vector_free(v);
	gsl_multimin_fminimizer_free(minimizer);
	printf("END WALK\n");
	
	int fIndex = peak.key / map->tCount;
	int tIndex = peak.key % map->tCount;
	// TODO return optimized parameters
	return TimeFreqValue{fIndex, tIndex, peak.value};
}

 */
