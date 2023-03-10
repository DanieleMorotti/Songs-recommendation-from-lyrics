"""
    This script creates the needed datasets, for the analysis in R and the GNN models, 
    taking the data from the MillionSongDataset http://millionsongdataset.com/ and 
    mergin these data with the complementary musiXmatch and Last.fm datasets that you
    can find at the same link.

    Usage:
        python create_dataset.py

    Author:
        Daniele Morotti
"""

import pandas as pd

import os, json
import urllib.request
from tqdm import tqdm
import shutil

import sqlite3

tqdm.pandas(desc="Working on the dataframe")


def tqdm_save_file(src, dest, length):
    '''
        Read the file from the urllib object and write in the destination path.
    ''' 
    chunk_size = 1024
    with tqdm(total=length, unit="B", unit_scale=True, desc="Downloading file") as pbar:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dest.write(chunk)
            pbar.update(len(chunk))


def download_files(files):
    '''
        The variable files has to be a list of tuples (url, dest_path).
    '''
    if not isinstance(files, list):
        print("ERROR: the passed value is not a list.")
        return None
    # Create the downloads directory if it doesn't exist
    if not os.path.exists('downloads/'):
        os.mkdir('downloads')

    for file in files:
        dest_path = os.path.join("downloads", file[1])
        # If the file has been already downloaded
        if os.path.exists(dest_path):
            print(f"The file {dest_path} is already present.")
            continue
        # Download and save the file
        with urllib.request.urlopen(file[0]) as src, open(dest_path, "wb") as dest:
            tqdm_save_file(src, dest, int(src.info()["Content-Length"]))
        # Unpack the file is necessary
        if file[1].endswith('.zip'):
            shutil.unpack_archive(dest_path, "downloads")

    return "All files saved correctly."


def prepare_mxm_dataset(file_path):
    '''
        Read the .txt file and retrieve the top 5000 words and
        the list of words for each song.
    '''
    with open(file_path, 'r') as fin:
        # In the given .txt the first lines are explanation of the data
        data = fin.readlines()[17:]
        top_words = data.pop(0)[1:].split(',')
        top_words = json.dumps(top_words)
        
        columns = ['track_id', 'mxm_id', 'words_count']

        ids = []
        mxm_ids = []
        word_counts = []
        # Iterate over all the rows to append the values to the dataframe
        for el in tqdm(data, desc="Building the dataset"):
            values = el.split(',')
            idx, counts = values[:2], values[2:]
            ids.append(idx[0])
            mxm_ids.append(idx[1])
            word_counts.append(counts)

        res_df = pd.DataFrame({columns[0]: ids, columns[1]: mxm_ids, columns[2]: word_counts})
    return top_words, res_df


def tuple_to_bow(val_list, words_list):
    '''
        It converts the mapping of <idx:count> to a list of words, where
        each word is repeated 'count' times.
    '''
    lyrics = ""
    for val in val_list:
        nums = val.split(':')
        # Take the relative word from the words list, the number of times it appears
        lyrics += f"{words_list[int(nums[0])-1]} " * int(nums[-1])
    return lyrics.rstrip()


def import_and_merge_tags(songs_data):
    '''
        It imports the tags and then it merges with the lyrics dataset.
    '''
    tags_conn = sqlite3.connect("downloads/lastfm_tags.db")
    # Take for each song the track id and the relative tag
    sql_query = "SELECT tids.tid as track_id, tags.tag FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND tid_tag.tid=tids.ROWID"
    tags_df = pd.read_sql_query(sql_query, tags_conn)
    tags_conn.close()

    print("\n- Adding the tags to the songs ...")
    full_df = pd.merge(songs_data, tags_df, on="track_id")
    full_df = full_df.drop_duplicates('track_id', ignore_index=True)

    return full_df


def create_complete_df(meta_path, df_data, words):
    '''
        It merges the data with the metadata, in order to have
        the title, artist, and the lyrics from the songs in the 
        same dataframe. Then it adds the tag of the genre for each song.
    '''
    mxm_meta = sqlite3.connect(meta_path)
    feat_to_keep = ['track_id', 'title', 'artist_name']

    df_meta = pd.read_sql_query("SELECT track_id, title, artist_name FROM songs", mxm_meta)

    full_df = pd.merge(df_data, df_meta, on="track_id")
    mxm_meta.close()

    # Create the lyrics
    tqdm.pandas(desc='Converting the word counts to lyrics')
    full_df['lyrics'] = full_df['words_count'].progress_apply(tuple_to_bow, words_list=words)
    full_df = full_df.drop(['mxm_id', 'words_count'], axis=1)

    # Add the tags
    full_df = import_and_merge_tags(full_df)

    return full_df


def save_file(data, dest_path, mode='csv', data_type='dataframe'):
    ''' 
        It saves the data into a file in dest_path. 

        Parameters:
            - data
                The data to save into a file.
            - dest_path: str
                The destination path of the file we want to save.
            - mode: str
                You can choose between 'json', 'txt' and 'csv'. If json passed
                the data will be dumped in json format.
            - data_type: str
                If 'dataframe' then the Pandas function to save as json
                will be called.
    '''
    if data_type == "dataframe":
        if mode == "json":
            data.to_json(dest_path)
        elif mode == "csv":
            data.to_csv(dest_path, index=False)
    else:
        if mode == 'json':
            with open(dest_path, 'w+') as fout:
                fout.write(json.dumps(data))
        elif mode == 'txt':
            # Transform the data to a string
            if not isinstance(data, str):
                data = json.dumps(data)
            with open(dest_path, 'w+') as fout:
                fout.write(data)
    

def aggregate_similar(str_list):
    '''
        It clean the data of the similar songs db, that gives the data in the
        following way:
            "id1,score1,id2,score2,..."
        and it returns a list of tuples:
            [(id1,score1), ...]
    ''' 
    sim_list = str_list.split(',')
    res = []
    for song, score in zip(sim_list[::2], sim_list[1::2]):
        res.append((song, float(score)))
    return res


def check_evaluation_dataset(target, track_id_list):
    '''
        It removes from the target string all the similar songs that are not
        in our main dataset.
    '''
    target_list = target.split(',')
    # We need to preserve the order of relevance for computing the mAP
    present = sorted(set(target_list).intersection(track_id_list), key=target_list.index)
    return ','.join(present)


def prepare_evaluation_dataset(songs_data):
    '''
        It prepares the evaluation dataset that contains for each song the 
        track_id and the relative list of similar songs. Actually it contains
        the string with the comma separated ids of similar songs.
    '''
    # Retrieved the data from the DB and put into a dataframe
    conn = sqlite3.connect("downloads/lastfm_similar_songs.db")
    similar_songs = pd.read_sql_query("SELECT tid as track_id, target FROM similars_src", conn)
    conn.close()

    # Merge the similar songs and take only the ones for which we have the lyrics
    evaluate_df = pd.merge(songs_data, similar_songs, on="track_id")
    
    tqdm.pandas(desc='Creating list of similar songs')
    reduced_df = evaluate_df.copy()
    reduced_df['target'] = reduced_df['target'].progress_apply(aggregate_similar)

    # Keep only the songs that have a list of similar ones with >=250 elements
    reduced_df = reduced_df[(reduced_df['target'].apply(len) >= 250)].reset_index(drop=True)

    # Remove the scores and keep only the ids of similar songs
    clean_list = lambda lis: ",".join([el[0] for el in lis])
    reduced_df['target'] = reduced_df['target'].apply(clean_list)

    # Keep only the target similar songs that are in the lyrics dataset.
    to_keep = songs_data.track_id #- set(reduced_df.track_id)
    tqdm.pandas(desc='Cleaning similar songs lists')
    reduced_df['target'] = reduced_df['target'].progress_apply(
                                        check_evaluation_dataset,
                                        track_id_list=to_keep
                                        )
    # Take at least 125 similar songs for each element
    reduced_df = reduced_df[reduced_df.target.str.split(',').apply(len) >= 125].reset_index(drop=True)
    # We need to remove also the ids of songs from the evaluation dataset, because
    # there will not be in the lyrics dataset.
    to_keep2 = set(songs_data.track_id) - set(reduced_df.track_id)
    reduced_df['target'] = reduced_df['target'].progress_apply(
                                        check_evaluation_dataset,
                                        track_id_list=to_keep2
                                        )

    reduced_df = reduced_df[reduced_df.target.str.split(',').apply(len) >= 106].reset_index(drop=True)
    
    # Save the file
    save_file(reduced_df, 'eval_similar_songs.csv', mode='csv', data_type='dataframe')

    return reduced_df, evaluate_df
    
    
def split_sim_scores_id(row, mode=1):
    ''' 
        It splits the similar ids and the relative scores in 2 separate columns.
    '''
    if mode==1:
        # Return the ids
        return ",".join(row.split(',')[::2])
    else:
        # Return the scores
        return ",".join(row.split(',')[1::2])


def check_gnn_dataset(row, track_id_list):
    '''
        It removes from the similar ids and scores all the the similar songs that are not
        in our main dataset.
    '''
    target_list = row['similars'].split(',')
    scores_list = row['sim_scores'].split(',')
    ind_func = target_list.index
    # We need to preserve the order of relevance for computing the mAP
    present = sorted(set(target_list).intersection(track_id_list), key=ind_func)
    
    # Retrieve only the scores kept by the search
    scores_to_keep = [ind_func(track) for track in present]
    pres_scores = list(map(scores_list.__getitem__, scores_to_keep))

    row['similars'] = ','.join(present)
    row['sim_scores'] = ','.join(pres_scores)
    return row


def create_gnn_dataset(similar_df, eval_df, val_size=0.2):
    '''
        It creates the training and validation datasets to use for the GNN.
    '''
    # Remove the songs that are in the test set
    gnn_df = similar_df[~similar_df['track_id'].isin(eval_df['track_id'])].reset_index(drop=True)
    # Split the similar data in 2 columns
    gnn_df['similars'] = gnn_df['target'].progress_apply(split_sim_scores_id, mode=1)
    gnn_df['sim_scores'] = gnn_df['target'].progress_apply(split_sim_scores_id, mode=2)
    # Remove the old column 'target'
    gnn_df = gnn_df.drop("target", axis=1)
    
    # Split in train and test 
    validation = gnn_df.sample(frac=val_size) 
    training = gnn_df.drop(validation.index)
    
    # Remove the songs that are not in the dataset from the similar one
    tqdm.pandas(desc="Cleaning the GNN dataset")
    training = training.progress_apply(check_gnn_dataset, axis=1, track_id_list=training['track_id'].tolist())
    validation = validation.progress_apply(check_gnn_dataset, axis=1, track_id_list=training['track_id'].tolist())
    
    # Remove the elements with less than 0 similar songs
    training = training[training['similars'].str.split(",").apply(len) > 0]
    validation = validation[validation['similars'].str.split(",").apply(len) > 0]
    
    save_file(training, 'training_songs_gnn.csv', mode='csv', data_type='dataframe')
    save_file(validation, 'validation_songs_gnn.csv', mode='csv', data_type='dataframe')

    

if __name__ == '__main__':
    # The urls from which you download the raw data
    train_url = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip"
    db_metadata_url = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db" 
    db_similar_url = "http://millionsongdataset.com/sites/default/files/lastfm/lastfm_similars.db"
    db_tags_url = "http://millionsongdataset.com/sites/default/files/lastfm/lastfm_tags.db"

    print("- Downloading the files ...")
    download_files([(train_url, "mxm_dataset.zip"), (db_metadata_url, "mxm_metadata.db"),
                    (db_similar_url, "lastfm_similar_songs.db"), (db_tags_url, "lastfm_tags.db")])

    print("\n- Retrieving the songs dataset and the top words ...")
    words, mxm_data_df = prepare_mxm_dataset("downloads/mxm_dataset_train.txt")
    save_file(words, 'top_5000_words.txt', mode="txt", data_type="list")
    words = json.loads(words)

    print("\n- Merge the data and create the lyrics ...")
    full_df = create_complete_df('downloads/mxm_metadata.db', mxm_data_df, words)

    print("\n- Building the evaluation dataset ...")
    eval_df, similar_df = prepare_evaluation_dataset(full_df)

    # Remove the songs to test from our "training" dataset and then save the dataset
    full_df = full_df[~full_df['track_id'].isin(eval_df['track_id'])].reset_index(drop=True)
    save_file(full_df, 'songs_data.csv', mode="csv", data_type="dataframe")
    
    print("\n- Building the dataset for the GNN models ...")
    create_gnn_dataset(similar_df, eval_df)

    print("\nProcess finished correctly!")

