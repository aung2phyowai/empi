#include <list>
#include <vector>

#include "envelope.hpp"
#include "fourier.hpp"

/**
 * @param C  number of channels
 * @param fourier  fourier transform coefficients for all C channels, corresponding to given frequency
 * @param z  equal to the ∫ f(t/s)² exp(2iωt) dt  where f(t) is the envelope function
 * @param w  frequency scaled to sampling frequency=1
 * @param sampleShift  offset between left edge and center of the envelope function
 * @param fit (optional) array of size C to be filled with computed parameters
 * @return
 */
inline double matchFrequency(FourierFrequencyView frequencyView, const complex& z, double w, double scale, double sampleShift, AtomFit* fit = nullptr) {
  double square = 0.0;
  double scaleSqrt = sqrt(scale);
  for (int c=0; c<frequencyView.C; ++c) {
    const complex v = frequencyView[c];
    double magnitude = std::abs(v);
    double phase = std::arg(v);
    /* TODO problem dla bardzo małych skal
     * bo a) optymalna faza zależy też od czynnika normalizacyjnego
     * b) normValue potrafi wyjść ujemny dla z~1 (!)
     */
    double normValue = 0.5 * (1 + std::real(z * std::polar(1.0, 2*phase))); // to można policzyć szybciej: sin/cos
    if (fit) {
      double phaseShift = 2*M_PI*std::remainder(w*sampleShift, 1.0);
      fit[c].amplitude = magnitude / normValue / scaleSqrt;
      fit[c].modulus = magnitude / std::sqrt(normValue);
      fit[c].phase = remainder(phase + phaseShift, 2*M_PI);
    }
    square += magnitude * magnitude / normValue;
  }
  return square;
}

static int roundForFFT(int x) {
  int y = 2;
  while (y < x) {
    y *= 2;
  }
  return y;
}

struct DictionaryBlock
{
    EnvelopeGenerator* const generator;
    const double scaleFactor;
    const double scale;
    const double scaleMin, scaleMax;

    double stepInFreq;
    int stepInTime;
    int samplesForFFT;
    int atomsInFreq;
    int atomsInTime;
    size_t atomsAll;

    Envelope envelope;
    std::vector<complex> envelopeFFT;

    DictionaryBlock(EnvelopeGenerator* generator, double scaleFactor, double scale, double scaleMin, double scaleMax)
      : generator(generator), scaleFactor(scaleFactor), scale(scale), scaleMin(scaleMin), scaleMax(scaleMax), fourierFreq(nullptr)
    { }

    void initialize(FourierFreq* fourierFreq)
    {
      this->fourierFreq = fourierFreq;
      // TODO poprawić bo przy gęstym słowniku będzie dużo takich samych planów
      FourierFull fourierFull(1, samplesForFFT, FFTW_ESTIMATE);
      FourierSpectrumView result = fourierFull.compute(samplesForFFT, &envelope.values, envelope.values, false, 0);
      envelopeFFT.resize(result.K); // TODO tu wystarczy do atomsInFreq chyba
      for (int k=0; k<result.K; ++k) {
        envelopeFFT[k] = result[k][0];
      }
    }

    /**
     * @param C  number of channels
     * @param fourier  fourier transform coefficients for all C channels, corresponding to given frequency
     * @param w  frequency ÷ sampling frequency
     * @param sampleShift  offset between left edge and center of the envelope function
     * @param fit (optional) array of size C to be filled with computed parameters
     * @return  sum of the 2nd moments of fits to all channels
     */
    double matchFrequencyValue(FourierFrequencyView frequencyView, double w, AtomFit* fit = nullptr) const {
      return matchFrequency(frequencyView, z(w), w, scale, envelope.shift, fit);
    }

    /**
     * @param C  number of channels
     * @param fourier  fourier transform coefficients for all C channels, corresponding to given frequency
     * @param k  index of frequency bin, should be between 0 and samplesForFFT/2
     * @param sampleShift  offset between left edge and center of the envelope function
     * @param fit (optional) array of size C to be filled with computed parameters
     * @return  sum of the 2nd moments of fits to all channels
     */
    double matchFrequencyBin(FourierFrequencyView frequencyView, int k, AtomFit* fit = nullptr) const {
      double w = static_cast<double>(k) / static_cast<double>(samplesForFFT);
      return matchFrequency(frequencyView, z(k), w, scale, envelope.shift, fit);
    }

private:
    FourierFreq* fourierFreq;

    complex z(int k) const {
      const int K = envelopeFFT.size();
      assert(K > 0);
      assert(k >= 0);
      assert(k < K);
      int k2 = -k*2;
      if (k2 < -K) {
        k2 += samplesForFFT;
      }
      complex z = (k2 >= 0) ? envelopeFFT[k2] : std::conj(envelopeFFT[-k2]);
      return z;
    }

    complex z(double w) const {
      assert(fourierFreq);
      return fourierFreq->computeNormFactor(envelope.values, w);
    }
};

class Dictionary
{
public:
    FourierFreq fourierFreqForEnvelope;
    const double energyError;
    const int sampleCount;

    const std::list<DictionaryBlock>& blocks;

    Dictionary(double energyError, int sampleCount)
      : fourierFreqForEnvelope(1), energyError(energyError), sampleCount(sampleCount), blocks(blocks_), atomsAll(0)
    {
      if (energyError <= 0 || energyError >= 1) {
        throw std::logic_error("energyError must be between 0 and 1");
      }
      if (sampleCount <= 0) {
        throw std::logic_error("signal length must be positive");
      }
    }

    void addBlocks(EnvelopeGenerator* generator, double scaleMin, double scaleMax, double frequencyMax = std::numeric_limits<double>::infinity())
    {
      double scaleFactor = generator->computeScaleFactor(1.0 - energyError);
      for (double scale = scaleMin * sqrt(scaleFactor); scale <= scaleMax; scale *= scaleFactor) {
        addBlock(generator, scaleFactor, scale, scaleMin, scaleMax, frequencyMax);
      }
    }

    void addBlock(EnvelopeGenerator* generator, double scaleFactor, double scale, double scaleMin, double scaleMax, double frequencyMax = std::numeric_limits<double>::infinity())
    {
      double root = std::sqrt(-2.0 / M_PI * std::log(1.0 - energyError));
      if (root >= 1) {
        throw Exception("dictionary is too sparse");
      }
      DictionaryBlock block(generator, scaleFactor, scale, scaleMin, scaleMax);
      generator->computeValues(scale, 0, block.envelope);

      double stepInTime = scale * root - 0.5;
      double samplesForFFT = scale / root + 0.5;
      assert(samplesForFFT > stepInTime);
      if (samplesForFFT >= EnvelopeGenerator::MAX_LENGTH) {
        throw Exception("maximal scale is too large");
      }
      if (stepInTime <= 1) {
        throw Exception("minimal scale is too small");
      }

      block.stepInTime = lrint(stepInTime);
      block.samplesForFFT = roundForFFT(std::max<int>(lrint(samplesForFFT), block.envelope.values.size()));
      block.stepInFreq = 1.0 / block.samplesForFFT;

      block.atomsInTime = 1 + (sampleCount - 1) / block.stepInTime;
      block.atomsInFreq = block.samplesForFFT / 2 + 1;
      if (std::isfinite(frequencyMax) && frequencyMax > 0) {
        block.atomsInFreq = std::min<int>(block.atomsInFreq, lrint(frequencyMax/block.stepInFreq - 0.5) + 1);
      }

      double atomsAll = static_cast<double>(block.atomsInTime) * static_cast<double>(block.atomsInFreq);
      if (atomsAll >= std::numeric_limits<size_t>::max()) {
        throw Exception("dictionary is too fine");
      }
      block.atomsAll = static_cast<size_t>(atomsAll);
      block.initialize(&fourierFreqForEnvelope);
      atomsAll += block.atomsAll;
      blocks_.push_back(std::move(block));
    }

    size_t getAtomCount() {
      return atomsAll;
    }

private:
    std::list<DictionaryBlock> blocks_;
    size_t atomsAll;
};
