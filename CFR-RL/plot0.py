import matplotlib.pyplot as plt

size = 500



x =[]
No_sr=[]
old_max = []
New_max = []




file_name = ''

for i in range(size):
    x.append(i)


file_H = 'PlotHeuristic.txt'
file_M = 'PlotMaxAction.txt'
file_R = 'PlotRandomAction.txt'
file_R1024 ='PlotRandomAction1024.txt'
file_M1024 = 'PlotMaxdomAction1024.txt'
file_RR1024 = 'PlotRAction1024.txt'
file_MM1024 = 'PlotMAction1024.txt'
file = open(file_MM1024,'r')

for j in range(size):
    line =file.readline().split()
    No_sr.append(float(line[0]))
    old_max.append(float(line[1]))
    New_max.append(float(line[2]))

file.close()

line =[]
label =[]

line.append(plt.plot(x, No_sr)[0])
label.append("No_SR")

line.append(plt.plot(x, old_max)[0])
label.append("old_max")

line.append(plt.plot(x, New_max)[0])
label.append("RAction1024")


plt.legend(line,label)

# plt.ylim(0, 3)
plt.xlabel('TM')
plt.ylabel('Utilization')
plt.show()


