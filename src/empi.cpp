/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <algorithm>
#include <map>
#include <fstream>
#include <iostream>
#include "conf.hpp"
#include "decomposition.hpp"
#include "envelope.hpp"
#include "io.hpp"
#include "timer.hpp"
#include "workspace.hpp"

static std::vector<int> parseIntegerSubset(const std::string& string, int max) {
	std::string scString;
	std::stringstream scStream(string);
	std::vector<int> numbers;
	while (std::getline(scStream, scString, ',')) {
		int start, end, status;
		status = sscanf(scString.c_str(), "%d-%d", &start, &end);

		if (status == 2 && start >= 1 && end <= max) {
			for (int i=start; i<=end; ++i) {
				numbers.push_back(i);
			}
		} else if (status == 1 && start >= 1 && start <= max) {
			numbers.push_back(start);
		} else {
			throw Exception("invalidSubsetSpecification");
		}
	}
	if (!numbers.size()) {
		throw Exception("emptySubsetSpecification");
	}
	if (static_cast<intmax_t>(numbers.size()) > static_cast<intmax_t>(std::numeric_limits<int>::max())) {
		throw Exception("tooLargeSubsetSpecification");
	}
	return numbers;
}

static void empi(const char* configFilePath) {
	std::unique_ptr<Dictionary> dictionary;
	std::unique_ptr<Decomposition> decomposition;
	std::unique_ptr<SignalReader> reader;
	std::shared_ptr<EnvelopeGenerator> generator;
	DecompositionSettings settings;

	// legacy configuration START /////////////////////////////////////
	Configuration legacyConfiguration;
	legacyConfiguration.parse(configFilePath);

	double energyError = atof(legacyConfiguration.at("energyError").c_str());
	if (energyError <= 0.0 || energyError >= 1.0) {
		throw Exception("invalidEnergyErrorValue");
	}
	double freqSampling = atof(legacyConfiguration.at("samplingFrequency").c_str());
	if (freqSampling <= 0.0) {
		throw Exception("invalidSamplingFrequency");
	}

	double scaleMin = legacyConfiguration.has("minAtomScale")
		? atof(legacyConfiguration.at("minAtomScale").c_str()) * freqSampling : 0.0;
	double scaleMax = legacyConfiguration.has("maxAtomScale")
		? atof(legacyConfiguration.at("maxAtomScale").c_str()) * freqSampling : INFINITY;
	double freqMax = legacyConfiguration.has("maxAtomFrequency")
		? atof(legacyConfiguration.at("maxAtomFrequency").c_str()) / freqSampling : INFINITY;

	generator.reset( new EnvelopeGeneratorTemplate<EnvelopeGauss>() );

	settings.iterationMax = atoi(legacyConfiguration.at("maximalNumberOfIterations").c_str());
	if (settings.iterationMax <= 0) {
		throw Exception("invalidMaximalNumberOfIterations");
	}
	settings.residualEnergy = 1 - 0.01 * atof(legacyConfiguration.at("energyPercent").c_str());
	if (settings.residualEnergy < 0 || settings.residualEnergy >= 1.0) {
		throw Exception("invalidEnergyPercentValue");
	}

	const std::string& pathToSignalFile = legacyConfiguration.at("nameOfDataFile");
	if (legacyConfiguration.has("numberOfSamplesInEpoch")) {
		int epochSize = atoi(legacyConfiguration.at("numberOfSamplesInEpoch").c_str());
		if (epochSize <= 0) {
			throw Exception("invalidNumberOfSamplesInEpoch");
		}
		if (legacyConfiguration.has("selectedEpochs")) {
			std::vector<int> epochs = parseIntegerSubset(legacyConfiguration.at("selectedEpochs"), std::numeric_limits<int>::max() / epochSize);
			if (epochs.empty()) {
				throw Exception("noSelectedEpochs");
			}
			reader.reset( new SignalReaderForSelectedEpochs(pathToSignalFile, epochSize, epochs) );
		} else {
			reader.reset( new SignalReaderForAllEpochs(pathToSignalFile, epochSize) );
		}
	} else {
		if (legacyConfiguration.has("selectedEpochs")) {
			// entry for numberOfSamplesInEpoch is missing
			legacyConfiguration.at("numberOfSamplesInEpoch");
		}
		reader.reset( new SignalReaderForWholeSignal(pathToSignalFile) );
	}

	reader->freqSampling = freqSampling;
	reader->channelCount = atoi(legacyConfiguration.at("numberOfChannels").c_str());
	if (reader->channelCount <= 0) {
		throw Exception("invalidNumberOfChannels");
	}
	reader->selectedChannels = parseIntegerSubset(legacyConfiguration.at("selectedChannels"), reader->channelCount);

	bool samePhase = false;
	int channelCount = reader->selectedChannels.size();
	int channelCountForBuilder = channelCount;
	std::string typeOfMP = legacyConfiguration.at("MP");
	std::transform(typeOfMP.begin(), typeOfMP.end(), typeOfMP.begin(), tolower);
	if (reader->selectedChannels.size() == 1 || typeOfMP == "smp") {
		decomposition.reset( new SmpDecomposition );
		channelCountForBuilder = 1;
	} else if (typeOfMP == "mmp1") {
		decomposition.reset( new Mmp1Decomposition );
		samePhase = true;
	} else if (typeOfMP == "mmp3") {
		decomposition.reset( new Mmp3Decomposition );
	} else {
		throw Exception("unsupportedDecompositionType");
	}

	// extracting base name of input file
	std::string prefix = reader->pathToSignalFile;
	std::string::size_type lastSlash = prefix.find_last_of('/');
	if (lastSlash != std::string::npos) {
		prefix = prefix.substr(lastSlash+1);
	}
	prefix = prefix.substr(0, prefix.find_last_of('.'));
	typeOfMP.resize(3);

	std::string pathToBookFile = legacyConfiguration.at("nameOfOutputDirectory")+"/"+prefix+"_"+typeOfMP+".b";

	// legacy configuration END ///////////////////////////////////////

	int epochProcessed = 0;
	BookWriter writer(pathToBookFile);
	std::unique_ptr<Workspace> workspace;
	while (true) {
		MultiSignal signal = reader->read();
		int sampleCount = static_cast<int>(signal.channels[0].samples.size());
		if (!sampleCount) {
			break;
		}
		if (!dictionary) {
			TIMER_START(prepareDictionary);
			dictionary.reset( new Dictionary(energyError, sampleCount) );
			dictionary->addBlocks(generator.get(), scaleMin, scaleMax, freqMax);
			workspace.reset( new Workspace(dictionary.get(), channelCountForBuilder, samePhase, true) );
			TIMER_STOP(prepareDictionary);
		}
		std::cout << "START" << '\t' << ++epochProcessed << '\t' << channelCount << '\t' << settings.iterationMax << '\t' << 100*(1-settings.residualEnergy) << std::endl;
		MultiChannelResult result = decomposition->compute(settings, workspace.get(), signal);
		writer.write(epochProcessed, signal, result);
	}
	writer.close();
	puts(" END");

	PRINT_TIMER(subtractAtom);
	PRINT_TIMER(subtractAtom_FFT);
	PRINT_TIMER(subtractAtom_OTHER);
	PRINT_TIMER(subtractAtom_COMPUTE);
	PRINT_TIMER(subtractAtom_UPDATE);
	PRINT_TIMER(subtractAtomFromSignal);
}

static void exception(const char* message) {
	std::cout << "ERROR\t" << message << std::endl;
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
	const char* configFilePath = 0;
	for (int i=1; i<argc; ++i) {
		if (*argv[i] != '-' && *argv[i] != 0) {
			configFilePath = argv[i];
		}
	}
	if (!configFilePath) {
		fprintf(stderr, "USAGE: %s path_to_config_file\n", argv[0]);
		return 1;
	}
	try {
		empi(configFilePath);
	} catch (const Exception& e) {
		exception(e.what());
	} catch (const std::bad_alloc& e) {
		exception("insufficientMemory");
	}
	exit(EXIT_SUCCESS);
}
