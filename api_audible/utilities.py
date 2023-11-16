import subprocess


def get_volume():
	# Get the current volume level using the pactl command
	# command = ['pactl', 'list', 'sinks']
	# result = subprocess.run(command, capture_output=True, text=True)

	# # Parse the output to get the current volume level of the active sink
	# output_lines = result.stdout.split('\n')
	# found_active = False
	# active_sink_volume_line = None
	# for s in output_lines:
	# 	# print(s)
	# 	if 'State: RUNNING' in s:
	# 		found_active = True
	# 	if found_active and ('Volume' in s):
	# 		active_sink_volume_line = s
	# 		break
	# if active_sink_volume_line is None:
	# 	# volume_level = -1.0
	# 	# exit()
	# 	return None, None
	# # Store the volume level as a floating point value between 0.0 and 1.0
	# active_sink_volume_str = active_sink_volume_line.split(' / ')[1].strip()
	# volume_level = int(active_sink_volume_str[:-1]) / 100.0
	# dB_str = active_sink_volume_line.split(' / ')[2].split('dB')[0]
	# dB = float(dB_str)
	dB = -20.0
	volume_level = 0
	return volume_level, dB
