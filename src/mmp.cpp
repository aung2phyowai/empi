/**********************************************************
 * University of Warsaw, Department of Biomedical Physics *
 *   Enhanced Matching Pursuit Implementation (empi)      *
 * See README.md and LICENCE for details.                 *
 **********************************************************/
#include "gabor.hpp"
#include "mmp.hpp"

MultiChannelResult Mmp1Decomposition::compute(const DecompositionSettings&, const WorkspaceBuilder&, const MultiSignal&) {
	throw std::logic_error("not supported yet"); // TODO
}

MultiChannelResult Mmp2Decomposition::compute(const DecompositionSettings&, const WorkspaceBuilder&, const MultiSignal&) {
	throw std::logic_error("not supported yet"); // TODO
}

MultiChannelResult Mmp3Decomposition::compute(const DecompositionSettings&, const WorkspaceBuilder&, const MultiSignal&) {
	throw std::logic_error("not supported yet"); // TODO
}
