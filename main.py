import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('50_Startups.csv')
df2 = df.copy()
state = pd.get_dummies(df2['State'],drop_first=True)
df2 = pd.concat([df2,state],axis=1)
df2.drop('State',axis=1,inplace=True)
df2.head()

sns.pairplot(df)
df2.head()


l = ['R&D Spend','Administration','Marketing Spend','Profit']
avg_list = []
range_list = []

for i in l:
    
    p = []
    avg = sum(df[i])/len(df[i])
    rang  = max(df[i]) - min(df[i])
    
    avg_list.append(avg)
    range_list.append(rang)
    
    for j in df[i]:
        p.append( (j-avg)/rang )
    
    df2[i] = p
    
profit = df2['Profit']
df2.drop('Profit',axis=1,inplace=True)
df2['Profit'] = profit
df2.head()

X1 = list(df2['R&D Spend'])
X2 = list(df2['Administration'])
X3 = list(df2['Marketing Spend'])
X4 = list(df2['Florida'])
X5 = list(df2['New York'])

Y  = list(df2['Profit'])

n = 10000      # no. of iterations
n1 = len(X1)  # no. of data points

p = []
a,b,c,d,f,g = 0,0,0,0,0,0
q = []

for j in range(n):
    
    r = 1.3
    sq_error = 0
    mean_sq = 0
    z1,z2,z3,z4,z5 = [],[],[],[],[]
    sq_error_list = []
    error_list = []
    predicted = []
    
    
    
    
    for i in range(n1) : 
        h = a + b*X1[i] + c*X2[i] + d*X3[i]+ f*X4[i]+ g*X5[i]
        e = h - Y[i]
        mean_sq += e**2
        error_list.append(e)
        sq_error_list.append(e**2)
        predicted.append(h)
        
        
        
    for k in range(n1) :
        z1.append(error_list[k]*X1[k])
        z2.append(error_list[k]*X2[k])
        z3.append(error_list[k]*X3[k])
        z4.append(error_list[k]*X4[k])
        z5.append(error_list[k]*X5[k])
    
    p.append(mean_sq/n1)
    q.append(j)
    
    # Weight Updating
    a -= r*(1/n1)*(sum(error_list))
    b -= r*(1/n1)*(sum(z1))
    c -= r*(1/n1)*(sum(z2))
    d -= r*(1/n1)*(sum(z3))
    f -= r*(1/n1)*(sum(z4))
    g -= r*(1/n1)*(sum(z5))

    print(f"Derivative Term : {round((1/n1)*(sum(z1)),4)}")
    print(f"Predicted : {round(h,4)} , Actual : {round(Y[n1-1],4)}")
    print(f"MSE :     {round((mean_sq)/(n1),7)} ")
    print(j)
    print("-"*100)
df2['Predicted']  = predicted

plt.plot(q ,p)

rnd = (1000.23 - avg_list[0]) / range_list[0]
admn = (124153.4  - avg_list[1]) / range_list[1]
mrkt = (1903.93 - avg_list[3])/range_list[2]
fl = 0 
ny = 1

pred = ( a + b*rnd + c*admn + d*mrkt + f*fl + g*ny )*range_list[-1] +  avg_list[-1]
print(pred)

predicted_2 = []
for i in predicted :
    z = i*range_list[-1] + avg_list[-1]
    predicted_2.append(z)
    
df['Predicted'] = predicted_2


SSres = []
SStot = []
avg = sum(list(df['Profit']))/n1

for i in range(n1) : 
    u = list(df['Predicted'])[i] - list(df['Profit'])[i]
    SSres.append(u**2)
    v = list(df['Profit'])[i] - avg
    SStot.append(v**2)
uu = sum(SSres)
vv = sum(SStot)
r_sq = 1- uu/vv
print(r_sq)
