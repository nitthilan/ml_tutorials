filename = "performance.txt"

with open(filename) as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

for line in content:
	words = line.split()
	if(len(words) and words[0]=="count=50"):
		print((float(words[5][4:])/1000)**2)
	if(len(words) and words[0]=="Graph:" and words[1].split("/")[4][:-8].split("_")[-1] == "2.0"):
		print(words[1].split("/")[4][:-8])


# 537.79929025
# 518.53665796
# 2625.07646025
# 2630.30508225
# 8114.87482276
# 8252.17812225
# 1319.64366361
# 1261.18737424
# 4776.86940201
# 4686.53883889
