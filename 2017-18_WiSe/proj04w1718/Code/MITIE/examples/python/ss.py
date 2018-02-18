with open('../../emerging.test.conll', 'r') as input:
	with open("newfile.txt","wb") as output: 
		for line in input:
			s = (line.rstrip('\n')).split('\t')
			if(s[0]!=''):
				if ((s[1]!='B-group')and(s[1]!='I-group')and(s[1]!='B-creative-work')and(s[1]!='I-creative-work')and(s[1]!='B-product')and(s[1]!='I-product')):
					output.write(line)
                
