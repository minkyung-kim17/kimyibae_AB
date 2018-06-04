import inspect, os, pickle, time

import pdb

current_path = inspect.getfile(inspect.currentframe())
current_dir = os.path.dirname(os.path.abspath(current_path))
EXP_PATH=os.path.join(current_dir,"experiences")

with open(os.path.join(EXP_PATH, 'replay_memoryAll'), 'rb') as f:	
		replay_memory = pickle.load(f)

start_time = time.time()

angles = [8, 10, 11, 14, 17, 18, 19, 20, 21, 22, 23, 26, 30, 31, 34, 35, 36, 46, 61, 65, 67, 70]
# angles = [8, 10, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 34, 35, 36, 41, 43, 46, 52, 55, 56, 58, 61, 63, 65, 67, 70, 72, 75]

taptimes = [600, 700, 900, 1000, 1100, 1200, 1300, 1500, 1600, 1700, 1800, 2000, 2500]

remake = []
for experience in replay_memory:
	if experience[1][0] in angles and experience[1][1] in taptimes: 
		remake.append(experience)

print("===%s seconds---" %(time.time()-start_time))
print(len(remake))

# 위에서 불렀던 파일에서 넣으면 안되는 각도 및 앵글 지움
remove_idx = 0
remove_experience = [[35, 1700], [46, 2500], [26, 1800]]
remove_reward = [57760, 68740, 67720]
for experience in remake:
	if experience[1] in remove_experience and experience[2] in remove_reward:
		print(remove_idx+1, experience)
		del remake[remove_idx]
	remove_idx += 1

# 지워졌나 확인
for experience in remake:
	if experience[1] in remove_experience and experience[2] in remove_reward:
		print(experience)
print(len(remake))
# pdb.set_trace()

# 파일로 저장
with open(os.path.join(EXP_PATH, 'replay_memory_%s'%time.strftime("%m%d_%H%M")), 'wb') as f:
	pickle.dump(remake, f)
# 0604_0057
print("Save...")

############################################
# pdb.set_trace()

# with open(os.path.join(EXP_PATH, 'replay_memory_0602_2342'), 'rb') as f:
# 		# pretrain_memory = pickle.load(f)
# 	replay_memory = pickle.load(f)

# for experience in replay_memory:
# 	if experience[1][0] == 55:
# 		print('55')
# pdb.set_trace()

