# Create cluster feature
kmeans = KMeans(n_clusters=6, #n_init= also)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] = X["Cluster"].astype("category")

X.head()



sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
);



X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6);