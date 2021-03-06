/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include <algorithm>
#include <cstdio>
#include <memory>
#include "classes.hpp"
#include "gabor.hpp"
#include "timer.hpp"

//------------------------------------------------------------------------------

void Workspace::subtractAtomFromSignal(Atom& atom, SingleSignal& signal, bool fit) {
	switch (atom.type) {
		case ATOM_GABOR:
			return GaborWorkspace::subtractAtomFromSignal(atom, signal, fit);
		default:
			throw Exception("invalidAtomGenerated");
	}
}

//------------------------------------------------------------------------------

MultiChannelResult Decomposition::compute(const DecompositionSettings& settings, Workspace* workspace, const MultiSignal& signal) {
	TIMER_START(compute);
	workspace->compute(signal);
	TIMER_STOP(compute);

	const int channelCount = signal.channels.size();
	MultiSignal residue(signal);
	MultiChannelResult result(channelCount);
	size_t atomCount = workspace->getAtomCount();
	const double totalEnergy = signal.computeEnergy();
	double residueEnergy = totalEnergy;
	if (totalEnergy == 0) {
		throw Exception("signalIsEmpty");
	}
	for (int iteration=1; iteration<=settings.iterationMax; ++iteration) {
		double energyProgress = 100.0 * (1.0 - residueEnergy / totalEnergy);
		double totalProgress = std::max(
			100.0 * (iteration - 1) / settings.iterationMax,
			energyProgress
		);
		std::cout << "ATOM\t" << (iteration - 1) << '\t' << atomCount
			<< '\t' << energyProgress << '\t' << totalProgress
			<< std::endl;

		TIMER_START(findBestMatch);
		Atoms bestMatches = workspace->findBestMatch();
		TIMER_STOP(findBestMatch);

		for (int c=0; c<channelCount; ++c) {
			result[c].push_back(bestMatches[c]);
		}
		if (iteration == settings.iterationMax) {
			break;
		}

		TIMER_START(subtractAtom);
		for (int c=0; c<channelCount; ++c) {
			workspace->subtractAtom(bestMatches[c], residue.channels[c], c);
		}
		TIMER_STOP(subtractAtom);

		residueEnergy = residue.computeEnergy();
		if (residueEnergy / totalEnergy <= settings.residualEnergy) {
			break;
		}
	}
	return result;
}

//------------------------------------------------------------------------------

MultiChannelResult SmpDecomposition::compute(const DecompositionSettings& settings, Workspace* workspace, const MultiSignal& signal) {
	MultiChannelResult result;
	int channelNumber = 0;
	for (const auto& channel : signal.channels) {
		MultiSignal wrapper;
		wrapper.channels.push_back(channel);
		std::cout << "CHANNEL\t" << channelNumber++ << std::endl;
		result.push_back(Decomposition::compute(settings, workspace, wrapper)[0]);
	}
	return result;
}
