#!/usr/bin/env python

"""
read flasher data
needs gcd generated with
gcdserver/resources/BuildGCD.py -r $run -o GCD.i3
"""
### IMPORT PACKAGES ###
from I3Tray import *
from icecube import dataio, dataclasses, payload_parsing
from icecube import DomTools, WaveCalibrator, wavedeform
import argparse


### PARSE ARGUMENTS ###
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
parser.add_argument('-f','--filelist', nargs='+', 
                    help='list of raw flasher data for one flashing DOM; Style: -f a b c', 
                    required=True)                           
parser.add_argument("-fl", "--filelocation", dest="filelocation", type=str,
                    help="location of the experimental flasher data (.i3 format)")
parser.add_argument("-f", "--filelist", dest="filelist", type=str)
# how to save:
parser.add_argument("-sl", "--savelocation", dest="savelocation", type=str)
parser.add_argument("-s", "--savename", dest="savename", type=str)
# gcd file:
parser.add_argument("-g", "--gcd", dest="gcd", type=str,
                    help="location and name of gcd file",
                    default="/data/exp/IceCube/2017/filtered/level2pass2/0101/Run00129004/Level2pass2_IC86.2016_data_Run00129004_0101_89_290_GCD.i3.zst")
parser.add_argument("-hdf", "--writeHDF", dest="writeHDF", type=bool,
                    help="if you want to save your file as HDF, not recommended",
                    default=False)
args = parser.parse_args()

GCDFILE = args.gcd

FILE_LIST = []
for i in args.filelist:
    FILE_LIST.append(args.filelocation+"/"+i)

SAVE = args.savelocation + args.savename

### ICETRAY ###
tray = I3Tray()

tray.Add(dataio.I3Reader, filenamelist=FILE_LIST)

tray.AddModule("QConverter", 'qify',WritePFrame=True)

tray.AddSegment(payload_parsing.I3DOMLaunchExtractor,
		MinBiasID = "MinBias",
		FlasherDataID = "Flasher",
		CPUDataID = "BeaconHits",
		SpecialDataID = "SpecialHits",

		# location of scintillators and IceACT
		SpecialDataOMs = [OMKey(0,1),
				  OMKey(12,65),
				  OMKey(12,66),
				  OMKey(62,65),
				  OMKey(62,66)]
		)

def findLeadingEdge(rawAtwdLst, atwdBinSize, ledLight):
    binNum = len(rawAtwdLst)
    initTime = 0
    if (ledLight > 30):
        threshold = 750

        # get bin index of awtd list first element whose pulse < 750
        index = 0
        for i,value in enumerate(rawAtwdLst):
            if (value < threshold):
                index = i
                break

        # Linear interpolation to determine exact time when threshold passed
        initTime = 0
        if (index > 0) and (index < binNum):
            initTime=((rawAtwdLst[index]-threshold)*(index-1.)+(threshold-rawAtwdLst[index-1])*(index))/(rawAtwdLst[index]-rawAtwdLst[index-1])

            if (initTime < 0):
                initTime = 0
            if (initTime > binNum):
                initTime = binNum - 1

        # Now compute the Leading Edge time
        sr = atwdBinSize
        leadEdgeTime = (4 + initTime) * sr
        return leadEdgeTime

    else:

        # compute the minimum pulse from the atwd vector and its index
        minIndex = 0
        minPulse = rawAtwdLst[0]

        for i,value in enumerate(rawAtwdLst):
            if (value < minPulse):
                minPulse = value
                minIndex = i

        # compute leading edge time
        minTime = float(minIndex)
        sr = atwdBinSize
        leadEdgeTime = (4 + initTime) * sr
        return leadEdgeTime

def flasherdecode(frame):
	cal = frame["I3Calibration"]
	status = frame["I3DetectorStatus"]
	domcal = cal.dom_cal
	domstatus = status.dom_status
	event_header = frame["I3EventHeader"]
	run = event_header.run_id
	subrun = event_header.sub_run_id
	runmap = frame["I3FlasherSubrunMap"][subrun]
	event_state = event_header.state

	flashervect = dataclasses.I3FlasherInfoVect()
	if (frame.Has("InIceFlasher") and event_state == 20):
		flashmap = frame["InIceFlasher"]

		for dom,launches in flashmap:
			for launch in launches:

				thisflasher=dataclasses.I3FlasherInfo()

				inspect.getdoc(launch)

				if str(launch.which_atwd) == "ATWDa":
					chip=0
				elif str(launch.which_atwd) == "ATWDb":
					chip=1

				thisflasher.atwd_bin_size = 1./dataclasses.atwd_sampling_rate(chip,domstatus[dom],domcal[dom])

				thisflasher.flashing_om = dom
				thisflasher.atwd_bin_size = 1./dataclasses.atwd_sampling_rate(chip,domstatus[dom],domcal[dom])
				thisflasher.raw_atwd3 = launch.raw_atwd[3]

				if dom in runmap:
					flashInfo = runmap[dom]
					thisflasher.led_brightness = flashInfo.brightness
					thisflasher.mask = flashInfo.mask
					thisflasher.width = flashInfo.window
					thisflasher.rate = flashInfo.rate

					leadingEdge = findLeadingEdge(launch.raw_atwd[3], thisflasher.atwd_bin_size, thisflasher.led_brightness)

					thisflasher.flash_time = launch.time + leadingEdge - 8.3
					flashervect.append(thisflasher)

		if (len(flashervect) > 0):
			frame["FlasherInfo"]=flashervect

tray.AddModule(flasherdecode, Streams=[icetray.I3Frame.DAQ])


def myfilter(frame):
	if (frame.Has("FlasherInfo")):
		flasherinfo = frame.Get("FlasherInfo")
		if (len(flasherinfo) != 1):
			return False
		else:
			return True
	else:
		return False

tray.AddModule(myfilter, Streams=[icetray.I3Frame.DAQ]) # If = lambda f: myfilter(f)

tray.AddModule("I3LCCleaning",
               InIceInput = "InIceRawData",
               InIceOutput = "InIceRawDataClean")

tray.AddModule("I3WaveCalibrator",
               Launches="InIceRawDataClean")

tray.AddModule("I3PMTSaturationFlagger")

tray.AddModule("I3Wavedeform")

def PulseShift(frame):
    writeFrame = False
    shiftedpulses = dataclasses.I3RecoPulseSeriesMap()
    if (frame.Has("WavedeformPulses") and frame.Has("FlasherInfo")):
        writeFrame = True
        pulse_map = frame["WavedeformPulses"]
        flashervect = frame.Get("FlasherInfo")
        for f in flashervect:
                ft = f.flash_time
        for om, pulse_series in pulse_map:
            vec = dataclasses.I3RecoPulseSeries()
            q_vect=[]
            t_vect=[]
            #print "OM Key: ", om
            for pulses in pulse_series:
                #print "Pulses: ", pulses        
                pulse = dataclasses.I3RecoPulse()
                q = pulses.charge
                t = pulses.time - ft
                pulse.time = t
                pulse.charge = q
                vec.append(pulse)
            shiftedpulses[om] = vec
    if (writeFrame):
	frame["FlasherShiftedPulses"] = shiftedpulses


tray.AddModule(PulseShift, Streams=[icetray.I3Frame.DAQ])

if args.writeHDF:
    from icecube import tableio,hdfwriter
    hdftable = hdfwriter.I3HDFTableService(sys.argv[-1])
    tray.AddModule(tableio.I3TableWriter,'hdf1', tableservice=hdftable,
                   SubEventStreams=['qify'],
                   keys=['FlasherShiftedPulses','FlasherInfo'])
else:
    tray.AddModule("I3Writer", "writeit",
                Filename = SAVE,
                Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics])


tray.AddModule("Dump")

tray.AddModule("TrashCan")
tray.Execute()
tray.Finish()
