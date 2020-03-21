
filename = "run_one.txt"

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