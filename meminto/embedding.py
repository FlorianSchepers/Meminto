#https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/speaker_verification.ipynb
from io import BytesIO
import os
from time import sleep
import numpy as np
import requests
import torchaudio

from meminto.audio_processing import SAMPLING_RATE, split_audio_dict
from meminto.transcriber import RemoteTranscriber, Transcriber
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as hierarchy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FILE_PATH = 'examples\multivoice.wav'

def main() -> None:

    transcriber = RemoteTranscriber(
            url=os.environ["TRANSCRIBER_URL"],
            authorization=os.environ["TRANSCRIBER_AUTHORIZATION"],
        )
    diarization = transcriber.diarize_audio(FILE_PATH)

    segmentes = diarization["segments"]

    diarization_list = []
    for segment in segmentes:
        diarization = {}
        diarization["start"] = segment["start"]
        diarization["end"] = segment["end"]
        diarization["text"] = segment["text"]
        diarization["speaker"] = "me"
        diarization["turn"] = segment["id"]
        diarization_list.append(diarization)
        print(f'turn: {segment["id"]}, start: {segment["start"]}, end: {segment["end"]}')
        print(segment["text"])
    
    print(len(diarization_list))
    sections = split_audio_dict(FILE_PATH, diarization_list)
    count = 0
    embeddings = []
    for audio_section in sections:
        buffer = BytesIO()
        torchaudio.save(
            buffer,
            audio_section.audio.unsqueeze(0),
            format="wav",
            sample_rate=SAMPLING_RATE,
        )
        buffer.seek(0)
        headers = {
            "Authorization": os.environ["TRANSCRIBER_AUTHORIZATION"],
        }
        files = {"audio_request": buffer}
        print(f"Endpoint used for embedding: https://speaker-embedding.model.tngtech.com/diarize?sample_rate=512000")
        response = requests.post(url='https://speaker-embedding.model.tngtech.com/diarize?sample_rate=512000', headers=headers, files=files)
        
        try:
            embeddings.append(response.json())
            count += 1
            
        except:
            print("skip")
            embeddings.append(embeddings[-1])
            pass
    #print(embeddings[0])
    # embeddings = np.array(embeddings)
    # embeddings = embeddings.reshape(-1, 1)
    # dist_matrix = distance.cdist(embeddings, embeddings, 'cosine')
    # condensed_dist_matrix = distance.squareform(dist_matrix)
    # linkage_matrix = hierarchy.linkage(condensed_dist_matrix, method='ward')
    # cluster_labels = hierarchy.cut_tree(linkage_matrix, n_clusters=None, height=0.5)
    # # clusters = hierarchy.fcluster(linkage_matrix, 0.5, criterion='distance')
    # print("Distance:")
    # print(cluster_labels)
    # print(len(cluster_labels))

    # embeddings = np.array(embeddings).reshape(5, -1)
    # kmeans = KMeans(n_clusters=3)
    # kmeans.fit(embeddings)
    # cluster_labels = kmeans.labels_

    #embeddings = np.array(embeddings).reshape(5, -1)  # Reshape to 2D array
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)

    silhouette_scores = []
    for n_clusters in range(2, 5):  # Try different numbers of clusters
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeddings)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append((n_clusters, silhouette_avg))

    # Find the number of clusters with the highest Silhouette Coefficient
    optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]

    kmeans = KMeans(n_clusters=optimal_n_clusters)
    kmeans.fit(embeddings)
    cluster_labels = kmeans.labels_.tolist()

    for indx, cluster_label in enumerate(cluster_labels):
        diarization_list[indx]["speaker"] = cluster_label

    print(cluster_labels)
    #print(len(embeddings))

    for segment in diarization_list:
        print(f'turn: {segment["turn"]}, speaker: {segment["speaker"]}, start: {segment["start"]}, end: {segment["end"]}')
        print(segment["text"])

if __name__ == "__main__":
    main()
