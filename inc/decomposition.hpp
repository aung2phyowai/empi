/**********************************************************
 * Piotr T. Różański (c) 2015–2018                        *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#ifndef EMPI_DECOMPOSITION_HPP
#define	EMPI_DECOMPOSITION_HPP

#include "base.hpp"
#include "workspace.hpp"

//------------------------------------------------------------------------------

struct DecompositionSettings {
	int iterationMax;
	double residualEnergy;
};

class Decomposition {
protected:
	Decomposition(bool samePhase)
		: samePhase(samePhase) { }

public:
	const bool samePhase;
	virtual ~Decomposition() =default;
	virtual MultiChannelResult compute(const DecompositionSettings& settings, Workspace* workspace, const MultiSignal& signal);
};

//------------------------------------------------------------------------------

class SmpDecomposition : public Decomposition {
public:
	SmpDecomposition(void) : Decomposition(false) { }

	MultiChannelResult compute(const DecompositionSettings& settings, Workspace* workspace, const MultiSignal& signal);
};

//------------------------------------------------------------------------------

class Mmp1Decomposition : public Decomposition {
public:
	Mmp1Decomposition(void) : Decomposition(true)
	{ }
};

//------------------------------------------------------------------------------

class Mmp3Decomposition : public Decomposition {
public:
	Mmp3Decomposition(void) : Decomposition(false)
	{ }
};

//------------------------------------------------------------------------------

#endif /* EMPI_DECOMPOSITION_HPP */
