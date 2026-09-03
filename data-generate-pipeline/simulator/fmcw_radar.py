import os
# Only set default GPU if not already specified (allows multi-GPU scripts to control GPU assignment)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import numpy as np


class FMCWRadar():
    def __init__(self, radar_cfg, start_global_time=0.0):
        self.cfg = radar_cfg
        self.start_global_time = start_global_time
        self.baseband_equivalent = True

        self.num_time_samples = int(radar_cfg.num_adc_samples / radar_cfg.adc_sample_rate * radar_cfg.sim_sample_rate)
        self.sampling_ratio = int(radar_cfg.sim_sample_rate / radar_cfg.adc_sample_rate)

    def generate_chirp(self):
        t = self._ts()
        return self._signal(t)

    def generate_mix_signal(self, tx_sig, rx_sig):
        tx_adc = self._adc_sampling(tx_sig)
        rx_adc = self._adc_sampling(rx_sig)
        mix_sig = tx_adc * torch.conj(rx_adc)
        return mix_sig

    def ideal_mix_signal(self, tau):
        if self.baseband_equivalent:
            f = self._adc_sampling(self.frequences()) + self.cfg.start_freq # real frequency
        else:
            f = self._adc_sampling(self.frequences())
        # Handle broadcasting for tau
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        f = f.unsqueeze(-1)  # [num_freqs, 1]
        tau = tau.unsqueeze(0)  # [1, num_tau]
        phase = 2 * np.pi * f * tau  # [num_freqs, num_tau]
        return torch.exp(torch.complex(
            torch.zeros_like(phase),
            phase))

    def frequences(self):
        t = self._ts()
        if self.baseband_equivalent:
            _freq = self.cfg.freq_slope * (t - self.cfg.idle_time)
        else:
            _freq = self.cfg.start_freq + self.cfg.freq_slope * (t - self.cfg.idle_time)
        return _freq

    def _signal(self, t):
        phase = 2 * np.pi * self._freq_t(t) * (t - self.cfg.idle_time)
        return torch.exp(torch.complex(
            torch.zeros_like(phase),
            phase))

    def _freq_t(self, t):
        if self.baseband_equivalent:
            _freq = self.cfg.freq_slope / 2 * (t - self.cfg.idle_time)
        else:
            _freq = self.cfg.start_freq + self.cfg.freq_slope / 2 * (t - self.cfg.idle_time)
        return _freq

    def _ts(self):
        return torch.linspace(
            start=self.cfg.adc_start_time,
            end=self.cfg.adc_sample_end_time,
            steps=self.num_time_samples + 1)[:-1]

    def _adc_sampling(self, sig):
        return sig[..., ::self.sampling_ratio]
