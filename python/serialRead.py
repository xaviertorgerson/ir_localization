import serial, sys

def decode(line): 
	output = line.strip('\n').strip('\r').split(',')
	#print(output)
	
	xx = int(output[0])
	yy = int(output[1])

	ww = int(output[2])
	zz = int(output[3])

	xxx = int(output[4])
	yyy = int(output[5])

	www = int(output[6])
	zzz = int(output[7])

	if(xx != 1023 or yy != 1023):
		print("Point 1 @ (x,y): ", xx, ",", yy)
	if(ww != 1023 or zz != 1023):
		print("Point 2 @ (x,y): ", ww, ",", zz)
	if(xxx != 1023 or yyy != 1023):
		print("Point 3 @ (x,y): ", xxx, ",", yyy)
	if(www != 1023 or zzz != 1023):
		print("Point 4 @ (x,y): ", www, ",", zzz)	


ser = serial.Serial(sys.argv[1], 19200)
print(ser.name)

while(True):
	line = ser.readline()
	decode(line)
ser.close()

