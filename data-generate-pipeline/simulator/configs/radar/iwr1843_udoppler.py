"""
Radar parameters
"""
light_speed = 3e8
start_freq = 77e9
wave_length = light_speed / start_freq

freq_slope = 70e12
adc_sample_rate = 5.209e6
num_adc_samples = 256

idle_time = 20.985e-6
adc_start_time = idle_time + 7e-6
ramp_end_time = idle_time + 57.14e-6
adc_sample_end_time = adc_start_time + num_adc_samples / adc_sample_rate
assert adc_sample_end_time < ramp_end_time, "Adc sampling time exceeds chirp duration"

num_chirps = 128
frame_period = 2e-2
chirp_last_period = ramp_end_time * num_chirps * 2
assert chirp_last_period <= frame_period, f"Duty circle exceeds frame period: {chirp_last_period} >= {frame_period}"

sim_sample_rate = adc_sample_rate

bandwidth = freq_slope * num_adc_samples / adc_sample_rate
range_resolution = light_speed / (2 * bandwidth)
doppler_resolution = wave_length / (2 * chirp_last_period)
max_range = num_adc_samples * range_resolution
max_doppler = num_chirps / 2 * doppler_resolution

print(f"Range resolution: {range_resolution} m")
print(f"Doppler resolution: {doppler_resolution} m/s")
print(f"Max range: {max_range} m")
print(f"Max doppler: {max_doppler} m/s")
