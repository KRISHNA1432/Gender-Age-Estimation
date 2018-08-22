import pandas as pd
import numpy as np
from sklearn import svm
'''
data = pd.read_csv('wiki5.csv')

x = data[data.columns[3:]]

y = data[data.columns[1:2]]

np.save('x.npy',x)
np.save('y.npy',y)
'''
print '20% complete'

x = np.load('x.npy')
y = np.load('y.npy')

row = x.shape[0]
row = int(0.75*row)

x_train = x[:row]
y_train = y[:row]

x_test = x[row:]
y_test = y[row:]

y_train = np.ravel(y_train)

print '30% complete'

model = svm.SVC(kernel='poly')
model.fit(x_train,y_train)

print '50% complete'

pred = model.predict(x_test)

acc = 0


print '80% complete'
for i in range(0,len(pred)):
    if pred[i] == y_test[i]:
        acc = acc + 1
        
acc = float(acc)/len(y_test)
acc = acc*100
print '100% complete'

print 'accuracy : ' , acc  , '%'
