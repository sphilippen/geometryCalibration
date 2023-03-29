#!/cvmfs/icecube.opensciencegrid.org/users/sphilippen/icetray/build/bin/python
#METAPROJECT /cvmfs/icecube.opensciencegrid.org/users/sphilippen/icetray/build
from I3Tray import *
from icecube import dataio, dataclasses, payload_parsing
from icecube import DomTools, WaveCalibrator, wavedeform, simprod
from icecube import icetray, dataclasses, dataio, phys_services, sim_services, clsim
from icecube import DOMLauncher
from icecube.simprod import segments
from icecube.sim_services import bad_dom_list_static
import argparse
load("sim-services")


# Parse Options:
parser = argparse.ArgumentParser(prog=("Detector simulation and calibration "
                                 "for flasher MCPE hits."),
                                 description=("Simulate MCPE hits for flasher " 
                                 "data with PPC in advanced using "
                                 "generate_flasher.py. This MCPE hits are "
                                 "often  directly used, but for comparison "
                                 "with experimental data this detector "
                                 "simulation and calibration is useful."),
                                 epilog="Questions? Contact Saskia Philippen")
# file to read in:                             
parser.add_argument("-fl", "--filelocation", dest="filelocation", type=str,
                    help="location where your MCPE file lies (.i3 format)")
parser.add_argument("-f", "--filename", dest="filename", type=str)
# how to save:
parser.add_argument("-sl", "--savelocation", dest="savelocation", type=str,
                    default="filelocation")
parser.add_argument("-s", "--savename", dest="savename", type=str)
# gcd file:
parser.add_argument("-g", "--gcd", dest="gcd", type=str,
                    help="location and name of gcd file",
                    default="/data/exp/IceCube/2017/filtered/level2pass2/0101/Run00129004/Level2pass2_IC86.2016_data_Run00129004_0101_89_290_GCD.i3.zst")
parser.add_argument("-k", "--keepMCPE", dest="keepMCPE", type=bool, 
                    default=False)

args = parser.parse_args()


GCDFILE = args.gcd
FILE = args.filelocation + args.filename
if args.savelocation == "filelocation":
    sl = args.filelocation
else: 
    sl = args.savelocation
SAVE = sl + args.savename





tray = I3Tray()

tray.Add(dataio.I3Reader,  FilenameList=[GCDFILE , FILE])

randomService = phys_services.I3SPRNGRandomService(seed = 1337, nstreams = 10, 
                                                   streamnum = 1)

tray.context['I3RandomService'] = randomService

def myfilter(frame):
        if (frame.Has("I3FlasherInfo")):
                flasherinfo = frame.Get("I3FlasherInfo")
                if (len(flasherinfo) != 1):
                        return False
                else:
                        return True
        else:
                return False

tray.AddModule(myfilter, Streams=[icetray.I3Frame.DAQ])

tray.AddSegment(segments.DetectorSim, 
                RandomService = 'I3RandomService', 
                GCDFile = GCDFILE, 
                InputPESeriesMapName = "MCPESeriesMap", 
                KeepMCHits = args.keepMCPE, 
                SkipNoiseGenerator = False,
                RunID=1)

tray.AddModule("I3LCCleaning",
               InIceInput = "InIceRawData",
               InIceOutput = "InIceRawDataClean")

tray.AddModule("I3WaveCalibrator", "sedan",
    Launches="InIceRawDataClean",  # fixed before I used InIceRawData
    Waveforms="CalibratedWaveforms",
    Errata="BorkedOMs",
    ATWDSaturationMargin=123,
    FADCSaturationMargin=0,
    )

tray.AddModule("I3PMTSaturationFlagger")

tray.AddModule('I3Wavedeform', 'deform')

def PulseShift(frame):
    if (frame.Has("WavedeformPulses") and frame.Has("FlasherInfo")):
	shiftedpulses = dataclasses.I3RecoPulseSeriesMap()
        pulse_map = frame["WavedeformPulses"]
        flashervect = frame.Get("FlasherInfo")
        for f in flashervect:
                ft = f.flash_time
        for om, pulse_series in pulse_map:
            vec = dataclasses.I3RecoPulseSeries()
            q_vect=[]
            t_vect=[]
            for pulse_org in pulse_series:       
                pulse = dataclasses.I3RecoPulse()
                pulse.time = pulse_org.time - ft
                pulse.charge = pulse_org.charge
                vec.append(pulse)
            shiftedpulses[om] = vec
	frame["FlasherShiftedPulses"] = shiftedpulses

tray.AddModule(PulseShift, Streams=[icetray.I3Frame.DAQ])

delKeys = ["CalibratedWaveforms", "InIceRawData", "InIceRawDataClean", 
           "WavedeformPulses"]


tray.AddModule("I3Writer", "writeit",
                Filename = SAVE,
                SkipKeys=delKeys,
                Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics])

tray.AddModule("TrashCan")
              
tray.Execute()
tray.Finish()              
