with open("submission2017","r") as input:
	x = 0
	with open("mtaner_output","w") as input2:    
		for line in input:
			s = line.rstrip('\n').split('\t')
			#print(len(s))
			if(len(s)>1 ):
			#	print(s[0]+' '+s[1])
				x = x+1
				print(x)
				input2.write(line)
               
