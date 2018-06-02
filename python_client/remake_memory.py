import inspect, os, pickle, time

import pdb

current_path = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(os.path.abspath(current_path))
EXP_PATH=os.path.join(current_dir,"experiences")

with open(os.path.join(EXP_PATH, 'replay_memoryAll'), 'rb') as f:	
		replay_memory = pickle.load(f)

start_time = time.time()

angles = [8, 10, 11, 14, 17, 18, 19, 20, 21, 22, 23, 26, 30, 31, 34, 35, 36, 46, 55, 61, 65, 67, 70]
# angles = [8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 34, 35, 36, 41, 43, 46, 52, 55, 56, 58, 61, 63, 65, 67, 70, 72, 75]

# taptimes = []

replay_memory_remake = []
for experience in replay_memory:
	if experience[1][0] in angles: # or experience[1][1] in taptimes: 
		replay_memory_remake.append(experience)

print("===%s seconds---" %(time.time()-start_time))

with open(os.path.join(EXP_PATH, 'replay_memory_%s'%time.strftime("%m%d_%H%M")), 'wb') as f:
	pickle.dump(replay_memory_remake, f)

print("Save...")

# pdb.set_trace()
