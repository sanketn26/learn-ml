from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from lib.course_data import load_customer_360

df = load_customer_360()
cols = ["mrr", "total_usage"]
_scaled = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(StandardScaler().fit_transform(df[cols]))


def task_1():
    return KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(df[cols].to_numpy()), _scaled


def task_3():
    return df.assign(cluster=_scaled).groupby("cluster")["is_churned"].mean()
