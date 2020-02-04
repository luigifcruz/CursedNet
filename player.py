#!/usr/bin/env python3

from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import SoapySDR
import signal
import queue
import torch
import numpy as np
import sounddevice as sd
from model import CursedNet

# Demodulator Settings
freq = 96.9e6
sfs = int(256e3)
afs = int(32e3)

sdr_buff = 1024
dsp_buff = sfs
dsp_out = int(dsp_buff/(sfs/afs))

# SoapySDR Configuration
args = dict(driver="airspyhf")
sdr = SoapySDR.Device(args)
sdr.setGainMode(SOAPY_SDR_RX, 0, True)
sdr.setSampleRate(SOAPY_SDR_RX, 0, sfs)
sdr.setFrequency(SOAPY_SDR_RX, 0, freq)

# Queue and Shared Memory Allocation
que = queue.Queue()
buff = np.zeros([dsp_buff], dtype=np.complex64)

# Load Neural model
model = CursedNet(input_ch=2, output_ch=1)
model.load_state_dict(torch.load('runs/PROTO1/model_save_epoch_25.pth'))
model.eval()


# Demodulation Function
def process(outdata, f, t, s):
    inp = que.get()
    inp = np.stack([np.real(inp), np.imag(inp)])
    inp = np.expand_dims(inp, axis=0).astype(np.float32)
    res = model(torch.tensor(inp))[0]
    outdata[:, 0] = res.detach().numpy()


# Graceful Exit Handler
def signal_handler(signum, frame):
    sdr.deactivateStream(rx)
    sdr.closeStream(rx)
    exit(-1)


signal.signal(signal.SIGINT, signal_handler)

# Start Collecting Data
plan = [(i*sdr_buff) for i in range(dsp_buff//sdr_buff)]
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx)

with sd.OutputStream(blocksize=dsp_out, callback=process,
                     samplerate=afs, channels=1):
    while True:
        for i in plan:
            sdr.readStream(rx, [buff[i:]], sdr_buff)
        que.put_nowait(buff.copy())