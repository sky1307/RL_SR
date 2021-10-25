import matplotlib.pyplot as plt

size = 50



x =[]
y =[]
No_sr=[]
H =[]
M =[]
R =[]



file_name = ''

for i in range(size):
    x.append(i)


file_H = 'PlotHeuristic.txt'
file_M = 'PlotMaxAction.txt'
file_R = 'PlotRandomAction.txt'

file_0 = open(file_H,'r')
file_1 = open(file_M,'r')
file_2 = open(file_R,'r')
for j in range(size):
    line_0 =file_0.readline().split()
    line_1 = file_1.readline().split()
    line_2 = file_2.readline().split()
    No_sr.append(float(line_0[0]))
    H.append(float(line_0[2]))
    M.append(float(line_1[2]))
    R.append(float(line_2[2]))

file_0.close()
file_1.close()
file_2.close()

line =[]
label =[]

line.append(plt.plot(x, No_sr)[0])
label.append("No_SR")

line.append(plt.plot(x, H)[0])
label.append("Heuristic")

line.append(plt.plot(x, M)[0])
label.append("MaxAction")

line.append(plt.plot(x, R)[0])
label.append("RandomAction")

plt.legend(line,label)

plt.xlabel('TM')
plt.ylabel('Utilization')
plt.show()


