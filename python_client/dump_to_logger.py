import pickle, logging

oneshotonekill_logger_mem = logging.getLogger("oneshotonekill_logger_mem")
# Debug < Info < Warning < Error < Critical
oneshotonekill_logger_mem.setLevel(logging.DEBUG)
oneshotonekill_logger_mem.addHandler(logging.StreamHandler())
oneshotonekill_logger_mem.addHandler(logging.FileHandler("oneshotonekill_logger_mem.log"))

oneshotonekill_memory = None
oneshotonekill_path = None
# with open("C:/Users/user/Documents/GitHub/kimyibae_AB/python_client/experiences_gathering/oneshotonekill_memory", "rb") as f:
# 	oneshotonekill_memory = pickle.load(f)
# for dirpath in oneshotonekill_memory:
# 	oneshotonekill_logger_mem.debug(dirpath)

with open("C:/Users/user/Documents/GitHub/kimyibae_AB/python_client/experiences_gathering/oneshotonekill_memory", "rb") as f:
	oneshotonekill_memory = pickle.load(f)
with open('C:/Users/user/Documents/GitHub/kimyibae_AB/python_client/experiences_gathering/oneshotonekill_path', 'rb') as f:
	oneshotonekill_path = pickle.load(f)
for i in range(len(oneshotonekill_path)):
	oneshotonekill_logger_mem.debug("%s [%d, %d] %d"%(oneshotonekill_path[i], oneshotonekill_memory[i][1][0], oneshotonekill_memory[i][1][1], oneshotonekill_memory[i][2]))
print('logger done')
