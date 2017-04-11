import matplotlib.pyplot as plt
import csv

av_x = 0
av_dx = 0
av_ddx = 0

def detectAnomaly(X):
    global av_x, av_dx, av_ddx

    predicted_X = []
    for i in range(len(X)):

        predicted_dx = av_dx + av_ddx
        predicted_x = av_x + predicted_dx
        predicted_X.append(predicted_x)

        x = X[i]
        if (i > 0): dx = X[i] - X[i-1]
        else: dx = 0

        if (i > 1): ddx = (X[i] - X[i-1]) - (X[i-1] - X[i-2])
        else: ddx = 0

        if i > 0: av_x += 0.3 * (x - av_x)
        else: av_x = x

        if i > 1: av_dx += 0.3 * (dx - av_dx)
        else: av_dx = dx

        if i > 2: av_ddx += 0.3 * (ddx - av_ddx)
        else: av_ddx = ddx

    return predicted_X

x = []
sol_t = []
sol = []
pol_t = []
pol = []
count = 0
with open('export_sol.csv') as csvfile:
    opmCounts = csv.reader(csvfile, quotechar='"')
    for row in opmCounts:
        for i in range(len(row)):
            row[i] = row[i].strip()
        print(row)
        x.append(count)
        sol_t.append(int(row[0]))
        sol.append(int(row[2]))
        count += 1

with open('export_pol.csv') as csvfile:
    opmCounts = csv.reader(csvfile, quotechar='"')
    for row in opmCounts:
        for i in range(len(row)):
            row[i] = row[i].strip()
        print(row)
        pol_t.append(int(row[0]))
        pol.append(int(row[2]))
        count += 1

sol_smooth = []
pol_smooth = []
for i in range(len(sol_t)):
    if i >= 3 and i <= len(sol_t) - 4:
         sol_smooth.append((sol[i-3] + sol[i-2] + sol[i-1] + sol[i] + sol[i+1] + sol[i+2] + sol[i+3])/7)
for i in range(len(pol_t)):
    if i >= 3 and i <= len(pol_t) - 4:
         pol_smooth.append((pol[i-3] + pol[i-2] + pol[i-1] + pol[i] + pol[i+1] + pol[i+2] + pol[i+3])/7)

#plt.plot(x,y)
del x[0]
del x[0]
del x[0]
del x[-1]
del x[-1]
del x[-1]
del sol_t[0]
del sol_t[0]
del sol_t[0]
del sol_t[-1]
del sol_t[-1]
del sol_t[-1]
del pol_t[0]
del pol_t[0]
del pol_t[0]
del pol_t[-1]
del pol_t[-1]
del pol_t[-1]

predicted_pol = detectAnomaly(pol_smooth)
plt.plot(sol_smooth[0:100])
plt.plot(pol_smooth[0:100])
plt.plot(predicted_pol[0:100])
plt.show()




    