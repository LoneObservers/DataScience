#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture 
from sklearn import metrics
from sklearn.model_selection import train_test_split
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import warnings
warnings.filterwarnings(action="ignore")
import os
print(os.listdir("../MMF1922"))



data= pd.read_csv("../MMF1922/CC GENERAL.csv")
data.drop('CUST_ID', axis=1, inplace=True) 
print(data.shape)
data.head()


print(data.describe())

print(data.corr())



plt.figure(figsize=(12,8),dpi=100)
sns.heatmap(data.corr(),cbar=True,square=True,annot=True,robust=True)
plt.show()


data.isnull().sum().sort_values(ascending=False)
#data.isna().mean()*100


data.dropna(subset=['CREDIT_LIMIT'], inplace=True)
data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].median(), inplace=True)



plt.figure(figsize=(20,35))
for i, col in enumerate(data.columns):
    if data[col].dtype != 'object':
        ax1 = plt.subplot(9, 2, i+1)
        sns.distplot(data[col], ax = ax1)
        plt.xlabel(col)
plt.show()



data1 = data.copy()
for c in data1:
    max3 = data1[c].median() + 3*data1[c].std()
    min3 = data1[c].median() - 3*data1[c].std()
    data1.loc[data1[c] > max3, c] = max3
    data1.loc[data1[c] < min3, c] = min3




X = np.asarray(data1)
scale = StandardScaler() 
X1 = scale.fit_transform(X)




### Centroid-based clustering: k-means
cost = []
for i in tqdm(range(2, 16)):
    kmean = KMeans(i)
    kmean.fit(X1)
    cost.append(kmean.inertia_)  
plt.figure(figsize=(12,8),dpi=80)
plt.plot(cost, 'bx-')
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()




kmean = KMeans(5)
kmean.fit(X1)
labels = kmean.labels_
clusters = pd.concat([data, pd.DataFrame({'cluster':labels})], axis=1)




for c in clusters:
    grid = sns.FacetGrid(clusters, col='cluster')
    grid.map(plt.hist, c)



dist = 1 - cosine_similarity(X1)
pca = PCA(2)
pca.fit(dist)
X_PCA = pca.transform(dist)



### visualization
x, y = X_PCA[:, 0], X_PCA[:, 1]

colors = {0: 'red',1: 'blue', 2: 'green', 3: 'yellow', 4: 'orange' }

###labels to be dicided
names = {0: 'who make all type of purchases', 1: 'more people with due payments',          2: 'who purchases mostly in installments', 3: 'who take more cash in advance',          4: 'who make expensive purchases'}
#,5:'who don\'t spend much money'}
  
df = pd.DataFrame({'x': x, 'y':y, 'label':labels}) 
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Customers Segmentation based on their Credit Card usage behaviour.")
plt.show()







### Connectivity-based clustering: Agglomerative hierarchical clustering
plt.figure(figsize =(6, 6)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X1, method ='ward'))) 



silhouette_scores = [] 

for n_cluster in tqdm(range(2, 11)):
    silhouette_scores.append(silhouette_score(X1, AgglomerativeClustering                                                (n_clusters = n_cluster).fit_predict(X1))) 

# Plotting a bar graph to compare the results 
plt.bar(range(2,21), silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 



agg = AgglomerativeClustering(n_clusters=3)
agg.fit(X1)
labels2 = agg.labels_
clusters2 = pd.concat([data, pd.DataFrame({'cluster':labels2})], axis=1)



for c in clusters2:
    grid = sns.FacetGrid(clusters2, col='cluster')
    grid.map(plt.hist, c)




colors2 = {0: 'red',1: 'blue', 2: 'green'}
#, 3: 'yellow', 4: 'orange',  5:'purple'}

###labels to be dicided
names2 = {0: 'who make all type of purchases', 1: 'more people with due payments',          2: 'who purchases mostly in installments'}
#,3: 'who take more cash in advance', 4: 'who make expensive purchases',5:'who don\'t spend much money'}
  
df2 = pd.DataFrame({'x':x, 'y':y, 'label':labels2}) 
groups2 = df2.groupby('label')

fig2, ax2 = plt.subplots(figsize=(20, 13)) 

for name, group in groups2:
    ax2.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors2[name],label=names2[name], mec='none')
    ax2.set_aspect('auto')
    ax2.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax2.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax2.legend()
ax2.set_title("Customers Segmentation based on their Credit Card usage behaviour.")
plt.show()



### Distribution-based clustering: Gaussian Mixture Models Clustering
def SelBest(arr, X):
    '''
    returns the set of X configurations with shorter distance
    '''
    dx = np.argsort(arr)[:X]
    return arr[dx]




### Silhouette score 
n_clusters = np.arange(2, 11)
sils = []
sils_err = []
iterations = 20
for n in tqdm(n_clusters):
    tmp_sil = []
    for _ in range(iterations):
        gmm = GaussianMixture(n, n_init=2).fit(X1) 
        labels = gmm.predict(X1)
        sil = metrics.silhouette_score(X1, labels, metric='euclidean')
        tmp_sil.append(sil)
    val = np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
    err = np.std(tmp_sil)
    sils.append(val)
    sils_err.append(err)



plt.figure(figsize=(12,8),dpi=80)
plt.errorbar(n_clusters, sils, yerr=sils_err)
plt.title("Silhouette Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Score")
plt.show()



def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)[0]
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)[0]
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return np.sqrt((log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2)



### Distance between GMMs
results = []
res_sigs = []
for n in tqdm(n_clusters):
    dist = []
    
    for iteration in range(iterations):
        train, test = train_test_split(X1, test_size=0.5)
        
        gmm_train = GaussianMixture(n, n_init=2).fit(train) 
        gmm_test = GaussianMixture(n, n_init=2).fit(test) 
        dist.append(gmm_js(gmm_train, gmm_test))
    selec = SelBest(np.array(dist), int(iterations/5))
    result = np.mean(selec)
    res_sig = np.std(selec)
    results.append(result)
    res_sigs.append(res_sig)



plt.figure(figsize=(12,8),dpi=80)
plt.errorbar(n_clusters, results, yerr=res_sigs)
plt.title("Distance between Train and Test GMMs", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Distance")
plt.show()



### Bayesian information criterion (BIC)
bics = []
bics_err = []
for n in n_clusters:
    tmp_bic = []
    for _ in range(iterations):
        gmm = GaussianMixture(n, n_init=2).fit(X1) 
        
        tmp_bic.append(gmm.bic(X1))
    val = np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))
    err = np.std(tmp_bic)
    bics.append(val)
    bics_err.append(err)



plt.figure(figsize=(12,8),dpi=80)
plt.errorbar(n_clusters,bics, yerr=bics_err, label='BIC')
plt.title("BIC Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Score")
plt.legend()




plt.figure(figsize=(12,8),dpi=80)
plt.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
plt.title("Gradient of BIC Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("grad(BIC)")
plt.legend()



gmm = GaussianMixture(n_components = 3) 
labels3 = gmm.fit_predict(X1)
clusters3 = pd.concat([data, pd.DataFrame({'cluster':labels3})], axis=1)




for c in clusters3:
    grid = sns.FacetGrid(clusters3, col='cluster')
    grid.map(plt.hist, c)





colors3 = {0: 'red',1: 'blue', 2: 'green'}
#, 3: 'yellow', 4: 'orange',  5:'purple'}

###labels to be dicided
names3 = {0: 'who make all type of purchases', 1: 'more people with due payments',          2: 'who purchases mostly in installments'}
#,3: 'who take more cash in advance', 4: 'who make expensive purchases',5:'who don\'t spend much money'}
  
df3 = pd.DataFrame({'x':x, 'y':y, 'label':labels3}) 
groups3 = df3.groupby('label')

fig3, ax3 = plt.subplots(figsize=(20, 13)) 

for name, group in groups3:
    ax3.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors2[name],label=names2[name], mec='none')
    ax3.set_aspect('auto')
    ax3.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax3.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax3.legend()
ax3.set_title("Customers Segmentation based on their Credit Card usage behaviour.")
plt.show()




