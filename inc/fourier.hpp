/**********************************************************
 * University of Warsaw, Department of Biomedical Physics *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_FOURIER_HPP
#define	EMPI_FOURIER_HPP

#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "fftw.hpp"

class FourierFrequencyView {
public:
    FourierFrequencyView(int C, const complex* data)
      : C(C), data(data)
    { }

    complex operator[](int c) const {
      assert(c >= 0);
      assert(c < C);
      return data[c];
    }

    int C;

private:
    const complex* data;
};

class FourierSpectrumView {
public:
    FourierSpectrumView(int K, int C, const complex* data)
      : K(K), C(C), data(data)
    { }

    FourierFrequencyView operator[](int k) const {
      assert(k >= 0);
      assert(k < K);
      return FourierFrequencyView(C, data+k*C);
    }

    int K, C;

private:
    const complex* data;
};

template<typename POINTER>
void ensure_same_phase(int C, POINTER values) {
	complex direction = 0;
	for (int c=0; c<C; ++c) {
		complex& value = values[c];
		direction += value * value;
	}
	double angle = 0.5 * std::arg(direction);
	//	commented version does not work, should be equivalent
	//	direction = std::polar(1.0, angle);
	for (int c=0; c<C; ++c) {
		complex& value = values[c];
		value = std::polar(std::abs(value) * std::cos(std::arg(value) - angle), angle);
		//value = std::abs(value) * std::real(value * direction) * direction;
	}
}

/**
 * Provides an FFTW-friendly allocator for standard containers (e.g. std::vector).
 * (Inspired by Stephen Lavavej's Mallocator.)
 * @tparam T  type of elements to be allocated, e.g. double for real-valued arrays
 */
template<class T>
class FourierAllocator {
public:
    typedef T value_type;

    FourierAllocator() noexcept { }

    template<class U>
    FourierAllocator(const FourierAllocator<U>&) noexcept
    { }

    template<class U>
    bool operator==(const FourierAllocator<U>&) const noexcept {
      return true;
    }

    template<class U>
    bool operator!=(const FourierAllocator<U>&) const noexcept {
      return false;
    }

    T* allocate(const size_t n) const {
      if (!n) return nullptr;
      if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::bad_array_new_length();
      }
      void* const result = fftw_malloc(n * sizeof(T));
      if (!result) throw std::bad_alloc();
      return static_cast<T*>(result);
    }

    void deallocate(T* const p, size_t) const noexcept {
      fftw_free(p);
    }
};

class FourierFull {
public:
	const int C, maxNfft;

private:
	std::vector<std::shared_ptr<fftw_plan_s>> plans;
  std::vector<real, FourierAllocator<real>> input;
  std::vector<complex, FourierAllocator<complex>> output;

	template<typename I>
	static I round2up(I N) {
		I Nfft = 1;
		while (Nfft < N) Nfft <<= 1;
		return Nfft;
	}

	template<typename I>
	static int iround2up(I N) {
		int index = 0;
		for (I Nfft=1; Nfft<N; Nfft<<=1) ++index;
		return index;
	}

public:
    FourierFull(int C, int maxNfft, unsigned flags)
      :
      C(C), maxNfft(maxNfft)
    {
      int maxKfft = (maxNfft / 2 + 1);
      input.resize(C * maxNfft);
      output.resize(C * maxKfft);
      for (int Nfft = 1; Nfft <= maxNfft; Nfft <<= 1) {
        plans.emplace_back(
          fftw_plan_many_dft_r2c(
            1,
            &Nfft,
            C,
            input.data(),
            nullptr,
            C,
            1,
            reinterpret_cast<fftw_complex*>(output.data()),
            nullptr,
            C,
            1,
            flags | FFTW_DESTROY_INPUT
          ),
          fftw_destroy_plan
        );
      }
    }

	template<class SIGNAL, class ENVELOPE>
	FourierSpectrumView compute(int Nfft, const SIGNAL& signal, const ENVELOPE& envelope, bool same_phase, int offset) {
		const int N = envelope.size();
		assert(N <= Nfft);
		assert(Nfft <= maxNfft);
		const int index = iround2up(Nfft);
		assert(index >= 0);
		assert(static_cast<unsigned>(index) < plans.size());
		assert(Nfft == 1<<index);
		const int Kfft = Nfft/2+1;
		const int Nstart = std::max(-offset, 0);
		const int Nend = std::min(N, static_cast<int>(signal[0].size()-offset));

    real *const input_ptr = input.data();
    complex *const output_ptr = output.data();
    if (Nstart > 0) {
      memset(input_ptr, 0, Nstart*C*sizeof(double));
    }
    real* input_sample = input_ptr + Nstart * C;
    for (int n=Nstart; n<Nend; ++n) {
      for (int c=0; c<C; ++c) {
        *input_sample++ = signal[c][offset+n] * envelope[n];
      }
    }
    if (Nend < Nfft) {
      memset(input_ptr+Nend*C, 0, (Nfft-Nend)*C*sizeof(double));
    }
    fftw_execute(plans[index].get());

    if (same_phase) {
      complex* output_sample = output_ptr;
      for (int k=0; k<Kfft; ++k) {
        ensure_same_phase(C, output_sample);
        output_sample += C;
      }
    }
    return FourierSpectrumView(Kfft, C, output_ptr);
	}
};

class FourierFreq {
	const int C;
	std::vector<complex> output;
	std::vector<complex> temp;

	template<class SIGNAL>
	void computeChannel(int c, const SIGNAL& signal, int offset, int Nstart, int Nend) {
		std::complex<double> result = 0.0;
		for (int n=Nstart; n<Nend; ++n) {
			result += signal[c][offset+n] * temp[n];
		}
		output[c] = result;
	}

	FourierFreq(const FourierFreq&) =delete;
	void operator=(const FourierFreq&) =delete;

public:
	FourierFreq(int C) :
		C(C), output(C)
	{ }

	template<class SIGNAL, class ENVELOPE>
	FourierFrequencyView compute(const SIGNAL& signal, const ENVELOPE& envelope, bool same_phase, int offset, double w) {
		return compute(signal, envelope, same_phase, offset, w, C);
	}

	template<class SIGNAL, class ENVELOPE>
  FourierFrequencyView compute(const SIGNAL& signal, const ENVELOPE& envelope, bool same_phase, int offset, double w, int C) {
    assert(C > 0);
    assert(C <= this->C);
		// w=1 represents sampling frequency
		complex *const output_ptr = output.data();
		const int N = envelope.size();
		temp.resize(N);
		const int Nstart = std::max(-offset, 0);
		const int Nend = std::min(N, static_cast<int>(signal[0].size()-offset));
		for (int n=0; n<Nstart; ++n) {
			temp[n] = 0;
		}
		for (int n=Nstart; n<Nend; ++n) {
			temp[n] = envelope[n] * std::polar(1.0, -2*M_PI*n*w);
		}
		for (int n=Nend; n<N; ++n) {
			temp[n] = 0;
		}

		if (C == 1) {
			computeChannel(0, signal, offset, Nstart, Nend);
		} else {
			for (int c=0; c<C; ++c) {
				computeChannel(c, signal, offset, Nstart, Nend);
			}
			if (same_phase) {
				ensure_same_phase(C, output_ptr);
			}
		}
		return FourierFrequencyView(C, output_ptr);
	}

  template<class ENVELOPE>
  complex computeNormFactor(const ENVELOPE& envelope, double w) {
    return compute(&envelope, envelope, false, 0, -2.0*w, 1)[0];
  }
};

#endif	/* EMPI_FOURIER_HPP */
