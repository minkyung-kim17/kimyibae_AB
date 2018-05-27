from py4j.java_gateway import JavaGateway, GatewayParameters

import pdb

class WrapperPython:
	# def __init__(self, ip, port=25555):
	def __init__(self, ip, port=20001):
		# ip = "127.0.0.1"
		self.java_gateway = JavaGateway(gateway_parameters=GatewayParameters(address=ip, port=port))

		# pdb.set_trace()

		self.entry_point = self.java_gateway.entry_point
		self.java_str = self.java_gateway.jvm.java.lang.String
		# *** ERROR:
		# py4j.protocol.Py4JNetworkError: An error occurred while trying to connect
		# to the Java server (localhost:25555)

	def get_slingshot(self, screenshot_path, silent=True):
		if not silent:
			print ("\tFind slingshot..."),
		slingshot_rect = self.entry_point.findSlingshot(screenshot_path)
		# x, y, width, height
		if slingshot_rect[0]==-1 or slingshot_rect[1]==-1:
			#sling point is not found. zoom out again
			print ("")
			print ("[ERROR]: sling point is not found. zoom out again or close the popup")
			return None
		if not silent:
			print ("found")
		# pdb.set_trace()
		return slingshot_rect

	# dqn_utils로 이동
	# def get_slingshot_refpoint(self, slingshot, silent=True):
	# 	X_OFFSET = 0.5
	# 	Y_OFFSET = 0.65
	# 	x = slingshot[0]+slingshot[2]*X_OFFSET
	# 	y = slingshot[1]+slingshot[2]*Y_OFFSET
	# 	return (x,y)

	# def get_blocks(self, screenshot_path, silent=True):
	# 	if not silent:
	# 		print ("\tFind blocks..."),
	# 	blocks_java = self.entry_point.findBlocks(screenshot_path, True)
	# 	if not silent:
	# 		print ("found %d blocks"%(len(blocks_java)))
	# 	blocks = []
	# 	for block_java in blocks_java:
	# 		block = []
	# 		for i in range(5):
	# 			block.append(block_java[i])
	# 		blocks.append(block)
	# 	return blocks
	#
	# def get_pigs(self, screenshot_path, silent=True):
	# 	if not silent:
	# 		print ("\tFind blocks..."),
	# 	pigs_java = self.entry_point.findPigs(screenshot_path, True)
	# 	if not silent:
	# 		print ("found %d blocks"%(len(pigs_java)))
	# 	pigs = []
	# 	for pig_java in pigs_java:
	# 		pig = []
	# 		for i in range(5):
	# 			pig.append(pig_java[i])
	# 		pigs.append(pig)
	# 	return pigs

	def save_seg(self, screenshot_path, save_path, silent=True):
		if not silent:
			print ("\tget segmented result..."),
		is_saved = self.entry_point.saveSegWithPath(screenshot_path, save_path)
		if not silent:
			if is_saved:
				print ("is saved")
			else:
				print ("failed")
		return is_saved

	# def get_birds(self, screenshot_path, silent=True):
	# 	if not silent:
	# 		print ("\tFind birds..."),
	# 	birds_java = self.entry_point.findBirds(screenshot_path, True)
	# 	if not silent:
	# 		print ("found %d birds"%(len(birds_java)))
	# 	birds = []
	# 	for bird_java in birds_java:
	# 		bird = []
	# 		for i in range(5):
	# 			bird.append(bird_java[i])
	# 		birds.append(bird)
	# 	return birds

	def get_score_in_game(self, screenshot_path):
		return self.entry_point.getScoreInGame(self.java_str(screenshot_path))
	def get_score_end_game(self, screenshot_path):
		return self.entry_point.getScoreEndGame(self.java_str(screenshot_path))
	# def get_score_game(self, screenshot_path, game_state):
	# 	return self.entry_point.getScoreGame(self.java_str(screenshot_path), self.java_str(game_state))
