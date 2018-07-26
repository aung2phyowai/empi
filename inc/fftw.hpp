/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_FFTW_HPP
#define	EMPI_FFTW_HPP

#include <complex>  // should be included before fftw3.h
#include <cstring>
#include <fftw3.h>

#include "base.hpp"

//------------------------------------------------------------------------------

/**
 * RAII-style wrapper for arrays allocated for FFTW.
 * @param T  type of values stored in array
 */
template<typename T>
class fftwArray {
	const int length;
	T* pointer = nullptr;

	fftwArray(const fftwArray&) =delete;
	void operator=(const fftwArray&) =delete;

public:
	/**
	 * Allocate array of given length.
	 * The contents of the array will remain uninitialized.
     * @param length  array's requested length (number of elements)
     */
	fftwArray(int length)
	: length(length) {
		pointer = static_cast<T*>(fftw_malloc(sizeof(T)*length));
		if (!pointer) {
			throw std::bad_alloc();
		}
	}

	fftwArray(fftwArray&& source)
	: length(source.length), pointer(source.pointer) {
		source.pointer = nullptr;
	}

	/**
	 * Destroy array by calling fftw_free on internal pointer.
     */
	~fftwArray() {
		if (pointer) fftw_free(pointer);
	}

	/**
	 * Return an internal pointer to the underlying data.
     * @return  C-style pointer to data
     */
	inline T* data(void) {
		return pointer;
	}

	/**
	 * Return i-th element of an array (counting from i=0).
	 * No range-checking is performed.
     * @param i  index of requested element
     * @return reference to i-th value
     */
	inline T& operator[](int i) {
		return pointer[i];
	}

	/**
	 * Return i-th element of an array (counting from i=0).
	 * No range-checking is performed.
     * @param i  index of requested element
     * @return constant reference to i-th value
     */
	inline const T& operator[](int i) const {
		return pointer[i];
	}

	/**
     * @return array's length (number of elements)
     */
	inline int size() const {
		return length;
	}

	/**
	 * Fill contents of the array with zeroes.
     */
	inline void zero() {
		memset(pointer, 0, sizeof(T)*length);
	}
};

//------------------------------------------------------------------------------

template<typename T>
class fftwArrayView {
	const int length;
	const T* const pointer;

public:
	fftwArrayView(int length, const T* pointer)
	: length(length), pointer(pointer)
	{ }

	/**
	 * Return an internal pointer to the underlying data.
     * @return  C-style pointer to data
     */
	inline const T* data(void) {
		return pointer;
	}

	/**
	 * Return i-th element of an array (counting from i=0).
	 * No range-checking is performed.
     * @param i  index of requested element
     * @return constant reference to i-th value
     */
	inline const T& operator[](int i) const {
		return pointer[i];
	}

	/**
     * @return array's length (number of elements)
     */
	inline int size() const {
		return length;
	}
};

//------------------------------------------------------------------------------

// convenience names for arrays allocated for FFTW
typedef fftwArray<double> fftwDouble;
typedef fftwArray<std::complex<double>> fftwComplex;
typedef fftwArrayView<std::complex<double>> fftwComplexView;

//------------------------------------------------------------------------------

/**
 * RAII-style wrapper for FFTW plan structure.
 */
class fftwPlan {
	std::shared_ptr<fftw_plan_s> plan;

	fftwPlan(const fftwPlan&) =delete;
	void operator=(const fftwPlan&) =delete;

public:
	/**
	 * Create plan of given length for real-to-complex transform.
	 * All parameters are analogous to fftw_plan_dft_r2c_1d specification.
	 */
	inline fftwPlan(int Nfft, real* input, complex* output, unsigned flags)
	: plan(
		fftw_plan_dft_r2c_1d(Nfft, input, reinterpret_cast<fftw_complex*>(output), flags),
		fftw_destroy_plan
	)
	{ }

	inline fftwPlan(fftwPlan&& source)
	: plan(std::move(source.plan))
	{ }

	/**
	 * Execute plan.
     */
	inline void execute() const {
		fftw_execute(plan.get());
	}

	/**
	 * Execute plan with different data arrays than it has been created for.
     */
	inline void execute(real* input, complex* output) const {
		fftw_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftw_complex*>(output));
	}
};

//------------------------------------------------------------------------------

/**
 * Return smallest integer power of 2 greater or equal than given value.
 * @param x  numeric value of any type
 * @return  smallest 2^n >= x in the same type as x
 */
template<typename T>
inline T fftwRound(T x) {
	T y = 2;
	while (y < x) {
		y *= 2;
	}
	return y;
}

#endif	/* EMPI_FFTW_HPP */
