from __future__ import division
import numpy
import os
from scipy.signal import lfilter
import first
from copy import deepcopy

winlen, ovrlen, pre_coef, nfilter, nftt = 0.025, 0.01, 0.97, 20, 512
ftThres = 0.5
vadThres = 0.4#(初始值0.4 )
opts = 1

filepath="/data/mileywang/cleanwav"

for root, directory, files in os.walk(filepath):
    for file in files:
        finwav=os.path.join(root,file)

        txt_file_name ="/data/mileywang/sp_result"
        fvad =txt_file_name+"/"+file.strip(".wav")+".txt"
        fs, data = first.speech_wave(finwav)
        ft, flen, fsh10, nfr10 = first.sflux(data, fs, winlen, ovrlen, nftt)

# --spectral flatness --
        pv01 = numpy.zeros(nfr10)
        try:
            pv01[numpy.less_equal(ft, ftThres)] = 1
            pitch = deepcopy(ft)
            pvblk = first.pitchblockdetect(pv01, pitch, nfr10, opts)
        except:
            nfr10 = nfr10 - 1
            pv01 = numpy.zeros(nfr10)
            pv01[numpy.less_equal(ft, ftThres)] = 1
            pitch = deepcopy(ft)
            pv01 = numpy.delete(pv01, [nfr10])
            # nfr10 = nfr10+1
            pvblk = first.pitchblockdetect(pv01, pitch, nfr10, opts)

# --filtering--
        ENERGYFLOOR = numpy.exp(-50)
        b = numpy.array([0.9770, -0.9770])#one demention filter 的选择
        a = numpy.array([1.0000, -0.9540])# one demention filter 的选择
        fdata = lfilter(b, a, data, axis=0)

        noise_samp, noise_seg, n_noise_samp = first.snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)

        for j in range(n_noise_samp):
            fdata[range(int(noise_samp[j, 0]), int(noise_samp[j, 1]) + 1)] = 0

        vad_seg = first.snre_vad(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)

        numpy.savetxt(fvad, vad_seg.astype(int), fmt='%i')
        print("%s --> %s " % (finwav, fvad))

        data = None;
        pv01 = None;
        pitch = None;
        fdata = None;
        pvblk = None;
        vad_seg = None



