from flucoma.fluid import mfcc, stats
from flucoma.utils import get_buffer
from pathlib import Path
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import umap
import click


json_data = {
    "projection" : []
}

@click.command()
@click.option('--numclusters', default=2, help="Number of clusters to find")

def analyse(numclusters):
    json_data["metadata"] = {
        "numclusters" : numclusters
    }
    source_path = Path("audio").expanduser().absolute()
    sources = [x for x in source_path.rglob("*.wav")]
    dataset = []

    for i, x in enumerate(sources):
        print(i)   
        d = get_buffer(stats(mfcc(x), numderivs=1))
        d = np.array(d).flatten() #flatten array
        dataset.append(d)
    
    dataset = np.array(dataset)

    std = StandardScaler().fit_transform(dataset)
    projection = umap.UMAP(n_components=3).fit_transform(std)
    projection_normalised = MinMaxScaler().fit_transform(projection)
    clustering = AgglomerativeClustering(n_clusters=numclusters).fit(projection)


    # now weave it into a fun loving dictionary
    for src, data, cluster in zip(sources, projection_normalised, clustering.labels_):
        json_data["projection"].append({
            "file" : str(src.name),
            "x" : float(data[0]),
            "y" : float(data[1]),
            "z" : float(data[2]),
            "cluster" : int(cluster)
        })

    with open('data.json', 'w') as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    analyse()