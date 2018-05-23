import socket, struct, binascii, time, io
from PIL import Image, ImageOps
#from common import *

import pdb

####################
#### MACROS
####################
MID_CONFIGURE=1

##### Query
MID_GETSTATE		= 12
MID_GETBESTSCORES	= 13
MID_GETCURRENTLEVEL	= 14
MID_DOSCREENSHOT 	= 11
MID_GETMYSCORE		= 23

#### Control
MID_CSHOOT		= 31
MID_PSHOOT		= 32
MID_SHOOTSEQ	= 33
MID_FULLYZOOMOUT= 34
MID_FULLYZOOMIN	= 35
MID_CLICKINCENTER = 36

#### Control (fast)
MID_CFASTSHOOT	= 41
MID_PFASTSHOOT	= 42
MID_SHOOTSEQFAST= 43

#### Operation
MID_LOADLEVEL	= 51
MID_RESTARTLEVEL= 52

####################
#### COMM APIs
####################

def hex_to_int(hex_str, is_reverse=True):
	if is_reverse:
		tokens = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
		# pdb.set_trace()
		try:
			reversed_hex = b''.join(list(reversed(tokens)))
			# print('done hex2int')
		except:
			print('error with hex_to_int')
			pdb.set_trace()
		# reversed_hex = ''.join(list(reversed(tokens)))
		return int(reversed_hex, 16)
	else:
		#in_order_hex = ''.join(tokens)
		return int(hex_str, 16)

def get_hex_MID(mid_int):
	return struct.pack('b', mid_int)

def comm_configure(s, client_id):
	# pdb.set_trace()
	print ("\t[OPERATION]: configure")
	s.sendall(bytes(get_hex_MID(MID_CONFIGURE)+struct.pack('!i', client_id)))
	data = s.recv(3)
	hex_str = binascii.hexlify(data)
	#s.close()

	# print('comfigure')
	# pdb.set_trace()
	round_info = hex_to_int(hex_str[:2])
	time_limit = hex_to_int(hex_str[2:4])
	num_of_levels = hex_to_int(hex_str[4:6])

	print ("\t\tround_info:", round_info)
	print ("\t\ttime_limit:", time_limit)
	print ("\t\tnum_of_levels:", num_of_levels)
	return round_info, time_limit, num_of_levels

def comm_click_in_center(s, silent=True):
	if not silent:
		print ("\t[OPERATION]: cartesian shoot (safe)"),
		print ("\t\tshooting params: (%d, %d, %d, %d, %d, %d)" % (x,y,dx,dy,t1,t2)),
	s.sendall(get_hex_MID(MID_CLICKINCENTER))
	data = s.recv(4)
	hex_str = binascii.hexlify(data)
	ret_status = hex_to_int(hex_str)
	if not silent:
		if ret_status == 1:
			print ("\t\treturn 1. clicking completed")
		else:
			print ("\t\treturn %d. clicking failed")
	return ret_status

def comm_c_shoot_safe(s, x, y, dx, dy, t1, t2, silent=True):
	if not silent:
		print ("\t[OPERATION]: cartesian shoot (safe)"),
		print ("\t\tshooting params: (%d, %d, %d, %d, %d, %d)" % (x,y,dx,dy,t1,t2)),
	s.sendall(get_hex_MID(MID_CSHOOT)+struct.pack('!i',x)+struct.pack('!i',y)+struct.pack('!i',dx)+struct.pack('!i',dy)+struct.pack('!i',t1)+struct.pack('!i',t2))
	data = s.recv(4)
	hex_str = binascii.hexlify(data)
	ret_status = hex_to_int(hex_str)
	if not silent:
		if ret_status == 1:
			print ("\t\treturn 1. shooting completed")
		else:
			print ("\t\treturn %d. shooting failed")
	return ret_status

def comm_c_shoot_fast(s, x, y, dx, dy, t1, t2, silent=True): # int 보내는거 이상함
	# pdb.set_trace()
	if not silent:
		print ("\t[OPERATION]: cartesian shoot (fast)"),
		print ("\t\tshooting params: (%d, %d, %d, %d, %d, %d)" % (x,y,dx,dy,t1,t2)),
	# s.sendall(get_hex_MID(MID_CFASTSHOOT)+struct.pack('!i',x)+struct.pack('!i',y)+struct.pack('!i',dx)+struct.pack('!i',dy)+struct.pack('!i',t1)+struct.pack('!i',t2))
	s.sendall(get_hex_MID(MID_CFASTSHOOT)+struct.pack('!i',int(x))+struct.pack('!i',int(y))+struct.pack('!i',int(dx))+struct.pack('!i',int(dy))+struct.pack('!i',int(t1))+struct.pack('!i',int(t2)))
	data = s.recv(4)
	hex_str = binascii.hexlify(data)
	ret_status = hex_to_int(hex_str)
	if not silent:
		if ret_status == 1:
			print ("\t\treturn 1. shooting completed")
		else:
			print ("\t\treturn %d. shooting failed")
	return ret_status

def comm_p_shoot_safe(s, x, y, r, theta, t1, t2, silent=True):
	# theta ranges from -9000~9000, the actual degree is divided by 100 (9025 means 90.25)
	if not silent:
		print ("\t[OPERATION]: polar shoot (safe)"),
		print ("\t\tshooting params: (%d, %d, %d, %d, %d, %d)" % (x,y,r,theta,t1,t2)),
	s.sendall(get_hex_MID(MID_PSHOOT)+struct.pack('!i',x)+struct.pack('!i',y)+struct.pack('!i',r)+struct.pack('!i',theta)+struct.pack('!i',t1)+struct.pack('!i',t2))
	data = s.recv(4)
	hex_str = binascii.hexlify(data)
	ret_status = hex_to_int(hex_str)
	if not silent:
		if ret_status == 1:
			print ("\t\treturn 1. polar shooting completed")
		else:
			print ("\t\treturn %d. polar shooting failed")
	return ret_status

def comm_p_shoot_fast(s, x, y, r, theta, t1, t2, silent=True):
	# theta ranges from -9000~9000, the actual degree is divided by 100 (9025 means 90.25)
	# horizontal is 0 degree. Positive angles shoots upward, negative angles shoots downward
	if not silent:
		print ("\t[OPERATION]: polar shoot (fast)"),
		print ("\t\tshooting params: (%d, %d, %d, %d, %d, %d)" % (x,y,r,theta,t1,t2)),
	s.sendall(get_hex_MID(MID_PFASTSHOOT)+struct.pack('!i',x)+struct.pack('!i',y)+struct.pack('!i',r)+struct.pack('!i',theta)+struct.pack('!i',t1)+struct.pack('!i',t2))
	data = s.recv(4)
	hex_str = binascii.hexlify(data)
	ret_status = hex_to_int(hex_str)
	if not silent:
		if ret_status == 1:
			print ("\t\treturn 1. polar shooting completed")
		else:
			print ("\t\treturn %d. polar shooting failed")
	return ret_status

def comm_click_in_center(s, silent=True):
	if not silent:
		print ("\t[OPERATION]: click in center"),
	s.sendall(get_hex_MID(MID_CLICKINCENTER))
	data = s.recv(4)
	hex_str = binascii.hexlify(data)
	ret_status = hex_to_int(hex_str)
	if not silent:
		if ret_status == 1:
			print ("\t\treturn 1. click finished")
		else:
			print ("\t\treturn %d. click failed")
	return ret_status

def comm_get_current_level(s, silent=True):
	if not silent:
		print ("\t[OPERATION]: comm get current level"),
	s.sendall(get_hex_MID(MID_GETCURRENTLEVEL))
	data = s.recv(4)
	hex_str = binascii.hexlify(data)
	ret_level = hex_to_int(hex_str)
	if not silent:
		print ("\t\tcurrent level: %d" % ret_level)
	return ret_level

def comm_load_level(s, level_num, silent=True):
	if not silent:
		print ("\t[OPERATION]: load level %d" % level_num),
	s.sendall(get_hex_MID(MID_LOADLEVEL)+struct.pack('b', level_num))
	data = s.recv(4)
	hex_str = binascii.hexlify(data)

	ret_status = hex_to_int(hex_str)
	if not silent:
		if ret_status == 1:
			# print('in comm.py')
			# pdb.set_trace()
			print ("\t\treturn 1. loaded level %d" % level_num)
		else:
			print ("\t\treturn %d. failed to load level %d" % level_num)
	return ret_status

def comm_get_state(s, silent=True):
	if not silent:
		print ("\t[OPERATION]: get state"),
	#s.connect((HOST,PORT))
	s.sendall(bytes(get_hex_MID(MID_GETSTATE)))
	data = s.recv(1)
	hex_str = binascii.hexlify(data)
	#s.close()

	game_state = hex_to_int(hex_str[:2])

	GAME_STATE={0:"UNKNOWN", 1:"MAIN_MENU", 2:"EPISODE_MENU", 3:"LEVEL_SELECTION", 4:"LOADING", 5:"PLAYING", 6:"WON", 7:"LOST"}

	if not silent:
		print ("\t\tgame_state:", game_state, GAME_STATE[game_state])
	return game_state

def comm_fully_zoomout(s, silent=True):
	if not silent:
		print ("\t[OPERATION]: fully zoomout"),
	s.sendall(bytes(get_hex_MID(MID_FULLYZOOMOUT)))
	data = s.recv(1)
	hex_str = binascii.hexlify(data)

	ret_status = hex_to_int(hex_str)
	if not silent:
		if ret_status == 1:
				print ("\t\treturn 1. fully zoomed out")
		else:
				print ("\t\treturn %d. failed to zoom out" % ret_status)
	
	return ret_status

def comm_fully_zoomin(s, silent=True):
	if not silent:
		print ("\t[OPERATION]: fully zoomin"),
	s.sendall(bytes(get_hex_MID(MID_FULLYZOOMIN)))
	data = s.recv(1)
	hex_str = binascii.hexlify(data)

	ret_status = hex_to_int(hex_str)
	if not silent:
		if ret_status == 1:
			print ("\t\treturn 1. fully zoomed in")
		else:
			print ("\t\treturn %d. failed to zoom in" % ret_status)
	
	return ret_status

def comm_get_best_scores(s, show_scores=True, silent=True):
	if not silent:
		print ("\t[OPERATION]: get best scores"),
	#s.sendall(bytes(get_hex_MID(MID_GETBESTSCORES)))
	s.sendall(bytes(get_hex_MID(MID_GETMYSCORE)))
	data = s.recv(1024)
	hex_str = binascii.hexlify(data)
	scores = []
	if len(data)==21*4:
		for i in range(21):
			scores.append(hex_to_int(hex_str[i*4:(i+1)*4]))

		if show_scores:
			if not silent:
				print ("\t\tscores:")
				for idx, score in enumerate(scores):
					print ("\t\t\t[LEVEL %d] %d"%(idx,score))

		return scores
	else:
		return None

def comm_do_screenshot(s, save_path=None, silent=True):
	# pdb.set_trace()
	print('do_screenshot')
	try:
		if not silent:
			print ("\t[OPERATION]: do screenshot"),
		s.sendall(bytes(get_hex_MID(MID_DOSCREENSHOT)))
		w_h_data = s.recv(4+4)
		hex_str = binascii.hexlify(w_h_data) # type: 'bytes'
		
		width = hex_to_int(hex_str[:8], is_reverse=False)
		height = hex_to_int(hex_str[8:16], is_reverse=False)
    
		# print(width)
		# print(height)
		# pdb.set_trace()

		image_data = b''
		while True:
			temp_data = s.recv(1024)
			if not temp_data:
				break
			image_data+=temp_data
			#print (len(image_data))
			if len(image_data)==width*height*3:
				break

		# pdb.set_trace()
		image = Image.frombuffer('RGB', (width, height), image_data, 'raw', 'RGB', 0, 1)
		# pdb.set_trace()
		if save_path:
			image.save(save_path)
		return image
	except Exception as e:
		print ("ERROR: exception occurred in do_screenshot")
		print (str(e))
		return None
		