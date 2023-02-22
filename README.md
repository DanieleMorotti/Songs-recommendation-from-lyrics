# Songs-recommendation-from-lyrics
A simple songs recommendation system implemented for the 'Text Mining' course at the University of Bologna.

<div align="center">
    <img src="/img/gnn.png" width="50%" />
    <p style="font-size:0.8rem" align="center">
        <em>GNN model that take the lyrics and returns the most similar songs</em> 
    </p>
</div>

The aim of this project is to first analyze the data of the songs taken from the [MillionSongDataset](http://millionsongdataset.com/) and merged with the [musiXmatch](http://millionsongdataset.com/musixmatch/) and the [Last.fm](http://millionsongdataset.com/lastfm/) extensions to get more features and then build a simple recommendation system based on the lyrics.

Firstly I tried to build a latent space, leveraging LSA, and then computing the cosine similarity between a vector of a song and all the others. This is a very simple approach and it performs poorly with a very low mAP (mean average precision) and recall. 

In a second notebook I tried to use the GNNs, that are more powerful and capable of learning relationships between the songs. I reduced the number of considered songs to avoid heavy computation and I reached very good results on this narrow subset. 

## Getting started
I suggest to run the notebooks on [Google Colab](https://colab.research.google.com/) to avoid problems with the libraries and with the resources needed to train.

Before starting, you should execute the script to download the raw data and create the needed datasets:
```python create_dataset.py```

after that you are ready to run the notebooks (if you are in Colab you only need to install the libraries using the cells that you will find in the notebooks).