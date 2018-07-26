/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <limits>
#include "io.hpp"

SignalReader::SignalReader(const std::string& pathToSignalFile)
: pathToSignalFile(pathToSignalFile) {
	file = fopen(pathToSignalFile.c_str(), "rb");
	if (!file) throw Exception("couldNotOpenSignalFile");
}

SignalReader::~SignalReader(void) {
	if (file) fclose(file);
}

MultiSignal SignalReader::readEpoch(int samplesToRead) {
	MultiSignal result;
	std::vector<float> sample(channelCount);
	const int C = selectedChannels.size();
	result.channels.resize(C);
	for (int i=0; i<C; ++i) {
		result.channels[i].freqSampling = freqSampling;
	}
	int leftToStore = std::numeric_limits<int>::max();
	while (samplesToRead-- > 0 && fread(sample.data(), sizeof(float), channelCount, file) == static_cast<size_t>(channelCount)) {
		if (--leftToStore < 0) {
			throw Exception("signalFileIsTooLongForThisMachine");
		}
		for (int i=0; i<C; ++i) {
			result.channels[i].samples.push_back(sample[selectedChannels[i]-1]);
		}
	}
	return result;
}

void SignalReader::seek(int sampleOffset) {
	if (fseek(file, sizeof(float) * channelCount * sampleOffset, SEEK_SET)) {
		throw Exception("signalSeekFailed");
	}
}

SignalReaderForAllEpochs::SignalReaderForAllEpochs(const std::string& pathToSignalFile, int epochSize)
: SignalReader(pathToSignalFile), epochSize(epochSize) { }

MultiSignal SignalReaderForAllEpochs::read() {
	MultiSignal result = readEpoch(epochSize);
	int sampleCount = result.channels[0].samples.size();
	if (sampleCount > 0 && sampleCount != epochSize) {
		throw Exception("fileIsTruncated");
	}
	return result;
}

SignalReaderForSelectedEpochs::SignalReaderForSelectedEpochs(const std::string& pathToSignalFile, int epochSize, const std::vector<int>& epochs)
: SignalReaderForAllEpochs(pathToSignalFile, epochSize) {
	for (int epoch : epochs) {
		this->epochs.push(epoch);
	}
}

MultiSignal SignalReaderForSelectedEpochs::read() {
	if (epochs.empty()) {
		return readEpoch(0);
	} else {
		int epoch = epochs.front();
		epochs.pop();
		seek(epochSize * (epoch-1));
		MultiSignal result = readEpoch(epochSize);
		int sampleCount = result.channels[0].samples.size();
		if (sampleCount != epochSize) {
			throw Exception("fileIsTruncated");
		}
		return result;
	}
}

SignalReaderForWholeSignal::SignalReaderForWholeSignal(const std::string& pathToSignalFile)
: SignalReader(pathToSignalFile) { }

MultiSignal SignalReaderForWholeSignal::read() {
	return readEpoch(std::numeric_limits<int>::max());
}

BookWriter::BookWriter(const std::string& pathToBookFile)
: pathToBookFile(pathToBookFile) {
	file = fopen(pathToBookFile.c_str(), "wb");
	if (!file) throw Exception("couldNotCreateOutputFile");
}

BookWriter::~BookWriter(void) {
	close();
}

void BookWriter::close(void) {
	if (file) {
		fclose(file);
		file = nullptr;
	}
}

void BookWriter::write(int epochNumber, const MultiSignal& signal, const MultiChannelResult& result) const {
	BookDataHeader headerData;
	BookDataSignalHeader headerSignal;
	BookDataAtomsHeader headerAtoms;
	BookDataAtomHeader headerAtom;

	const int C = signal.channels.size();
	const int N = C ? signal.channels.front().samples.size() : 0;

	if (epochNumber == 1) {
		BookHeader headerStart;
		// TODO meaningful information
		setBE(headerStart.content.channelCount, C);
		setBE(headerStart.content.dictionarySize, 0);
		setBE(headerStart.content.energyPercent, 100.0);
		setBE(headerStart.content.iterationCount, 1);
		setBE(headerStart.content.pointsPerMicrovolt, 1.0);
		setBE(headerStart.content.freqSampling, C ? signal.channels.front().freqSampling : 0.0);
		fwrite(&headerStart, sizeof headerStart, 1, file);
	}

	long headerDataPosition = ftell(file);
	setBE(headerData.epochNumber, epochNumber);
	setBE(headerData.sampleCount, N);
	fwrite(&headerData, sizeof headerData, 1, file);

	std::vector<float> sampleBuffer(N);
	std::vector<float> paramsBuffer(6);
	for (int c=0; c<C; ++c) {
		setBE(headerSignal.channelNumber, c+1);
		setBE(headerSignal.len, sizeof headerSignal.channelNumber + sizeof(float) * N);
		fwrite(&headerSignal, sizeof headerSignal, 1, file);
		for (int i=0; i<N; ++i) {
			setBE(sampleBuffer[i], signal.channels[c].samples[i]);
		}
		fwrite(sampleBuffer.data(), sizeof(float), N, file);

		long headerAtomsPosition = ftell(file);
		setBE(headerAtoms.channelNumber, c+1);
		fwrite(&headerAtoms, sizeof headerAtoms, 1, file);
    const size_t P = 6;
    for (const AtomResult& atom : result[c]) {
      setBE(paramsBuffer[0], atom.modulus);
      setBE(paramsBuffer[1], atom.amplitude);
      setBE(paramsBuffer[2], atom.center);
      setBE(paramsBuffer[3], atom.scale);
      setBE(paramsBuffer[4], atom.frequency*2.0);
      setBE(paramsBuffer[5], atom.phase);
      setBE(headerAtom.len, sizeof(float) * P);
      setBE(headerAtom.type, 13);
      fwrite(&headerAtom, sizeof headerAtom, 1, file);
      fwrite(paramsBuffer.data(), sizeof(float), P, file);
    }

		long currentPosition = ftell(file);
		fseek(file, headerAtomsPosition, SEEK_SET);
		setBE(headerAtoms.len, currentPosition - headerAtomsPosition - 5);
		fwrite(&headerAtoms, sizeof headerAtoms, 1, file);
		fseek(file, currentPosition, SEEK_SET);
	}

	long currentPosition = ftell(file);
	fseek(file, headerDataPosition, SEEK_SET);
	setBE(headerData.len, currentPosition - headerDataPosition - 5);
	fwrite(&headerData, sizeof headerData, 1, file);
	fseek(file, currentPosition, SEEK_SET);
}
