import numpy as np
import matplotlib.pyplot as plt


fig=plt.figure(figsize=(8,6))
#Generating a Gaussion dataset:
#creating random vectors from the multivariate normal distribution
#given mean and covariance
mu_vec1=np.array([0,0])
cov_mat1=np.array([[1,0],[0,1]])
X=np.random.multivariate_normal(mu_vec1,cov_mat1,500)
R=X**2
R_sum=R.sum(axis=1)
plt.scatter(X[:,0],X[:,1],color='green',marker='o',
            s=32.*R_sum,edgecolor='black',alpha=0.5)


plt.show()