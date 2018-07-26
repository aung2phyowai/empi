/**********************************************************
 * University of Warsaw, Department of Biomedical Physics *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include "envelope.hpp"

void assertEquals(double expected, double actual) {
	double diff = fabs(actual - expected);
	if (diff > 1.0e-10) {
		printf("ERROR: diff = %lf\n", diff);
		exit(1);
	}
}

void test(EnvelopeGenerator* generator, double scale, double center) {
	Envelope envelope;
	generator->computeValues(scale, center, envelope);

	// testing energy
	double actual = 0.0;
	for (double value : envelope.values) {
		actual += value * value;
	}
	assertEquals(1.0, actual);

	// testing center
	assertEquals(center, envelope.offset + envelope.shift);

	// testing values
	int size = envelope.values.size();
	for (int i=0; i<size; ++i) {
		double x = envelope.offset + i;
		double value = EnvelopeGauss::computeValue((x - center) / scale) / sqrt(scale);
		assertEquals(value, envelope.values[i]);
	}

  // special test for center=0
  if (center == 0) {
    int sizeHalf = size / 2;
    assertEquals(sizeHalf*2+1, size);
    assertEquals(1.0 / sqrt(scale), envelope.values[sizeHalf]);
    assertEquals(sizeHalf, envelope.shift);
    assertEquals(-sizeHalf, envelope.offset);
  }
}

int main(void) {
	EnvelopeGeneratorTemplate<EnvelopeGauss> generator;
  test(&generator, 30.0, 0.0);
	test(&generator, 100.0, 0.0);
  test(&generator, 101.4, 0.0);
	test(&generator, 200.0, 5.5);
	test(&generator, 50.0, -31.7);
	test(&generator, 150.0, 13.0);
}
