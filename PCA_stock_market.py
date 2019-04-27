import pandas as pd
import plotly
import plotly.plotly as py
import matplotlib.pyplot as plt

plotly.tools.set_credentials_file(username='ksoumya', api_key='uFWAHNIiAwtCsLXcxdGI')
df = pd.read_csv('C:/Users/ADMIN/Downloads/outclose.csv')

df.columns=['Price', 'Open', 'High', 'Low',	'OBV', 'VPT', 'GOLD', 'FEX', 'OIL','PredClose']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()

X = df.iloc[:,0:9].values
y = df.iloc[:,9].values

xc =X[:,0]


plt.scatter(xc,y)
plt.ylabel('dependent variable')
plt.xlabel('independent variable')
plt.show()



from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

print(var_exp)
x = ['PC %s' %i for i in range(1,10)]

x = np.array(x)
print(x)
print(cum_var_exp)

trace1 = plt.bar(x, height=var_exp)

trace2 = plt.scatter(x, cum_var_exp)


data = [trace1, trace2]


matrix_w = np.hstack((eig_pairs[0][1].reshape(9,1),
                      eig_pairs[1][1].reshape(9,1)))

print('Matrix W:\n', matrix_w)


Y = X_std.dot(matrix_w)
len(Y)

for i in range(len(Y)):
    plt.scatter(Y[i,0],Y[i,1])
    plt.ylabel('dependent variable')
    plt.xlabel('independent variable')
plt.show()


