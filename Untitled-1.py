s="barfoobarfoofoobar"
words=["foo","bar"]
output=[]
for i in range(len(s)):
    if i in words:
        output.append(i.index())
        print(output)

