# How to process with simulated and experimental data

# Simulation
- `generateMCPE.py` simulates MCPE hit using icetray's PPC, be aware to configure PPC in such the way that you use the correct ice model.
- `detectorSimulation.py` takes MCPE hits and performes the simulation of detector effects and processes the simulation data up to wavedeform 
in the same way the experimental data are handled.

# Expermintal Data
- processFlasherExperimental.py takes a list of raw data from a single flashing DOM and performes the processing and save wavedeform pulses
