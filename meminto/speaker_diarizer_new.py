from dataclasses import dataclass
from click import Tuple
import torch
import numpy as np
from transformers import pipeline
from pyannote.audio import Inference
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict, Any, Optional, Union
from torch import Tensor

class SpeakerDiarizer:
    def __init__(
        self,
        embedding_model_name: str = "pyannote/embedding",
        device: Optional[Union[str, torch.device]] = None,
        hugging_face_auth_token: Optional[str] = None,
    ) -> None:
        """
        Initialize the SpeakerDiarizer with the specified embedding model.

        :param embedding_model_name: Name of the speaker embedding model.
        :param device: Device to run the model on ('cuda' or 'cpu').
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.embedding_model_name = embedding_model_name
        self.embedding_model = Inference(
            model=embedding_model_name,
            device=device,
            use_auth_token=hugging_face_auth_token,
        )

    def compute_embeddings(
        self,
        audio_segments: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Compute embeddings for each audio segment.

        :param audio_segments: List of audio segments with metadata.
        :return: List of embeddings with associated metadata.
        """
        embeddings: List[np.ndarray] = []

        for segment in audio_segments:
            audio_tensor: Tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(
                0
            )
            with torch.no_grad():
                embedding = self.embedding_model(
                    {"waveform": audio_tensor, "sample_rate": 16000}
                )
                embedding_data: np.ndarray = (
                    embedding.data
                )  # Shape: (num_frames, embedding_dim)

                # Handle cases where there are multiple frames
                if embedding_data.shape[0] == 1:
                    # Only one frame; use it directly
                    embedding_vector: np.ndarray = embedding_data[0]
                else:
                    # Multiple frames; average the embeddings
                    embedding_vector: np.ndarray = np.mean(embedding_data, axis=0)

            embeddings.append(embedding_vector)
        return embeddings

    def cluster_embeddings(
        self,
        embeddings: List[np.ndarray],
        num_speakers: Optional[int] = None,
    ) -> List[str]:
        """
        Cluster the embeddings to assign speaker labels.

        :param embeddings: List of embeddings with metadata.
        :param num_speakers: Optional number of speakers. If None, the number will be estimated.
        :return: List of embeddings with assigned speaker labels.
        """
        embedding_vectors: np.ndarray = np.array(embeddings)

        if num_speakers is None:
            num_speakers = self.estimate_num_speakers(embedding_vectors)
            print(f"Estimated number of speakers: {num_speakers}")

        clustering = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric="cosine",
            linkage="average",
        )

        clustering_labels: np.ndarray = clustering.fit_predict(embedding_vectors)

        speaker_labels = [f"Speaker {label}" for label in clustering_labels]

        return speaker_labels

    def diarize(
        self,
        audio_segments: List[np.ndarray],
        num_speakers: Optional[int] = None,
    ) -> List[str]:
        """
        Perform speaker diarization on the audio segments.

        :param audio_segments: List of audio segments with metadata.
        :param num_speakers: Optional number of speakers.
        :return: List of segments with assigned speaker labels.
        """
        embeddings = self.compute_embeddings(audio_segments)
        speaker_lables = self.cluster_embeddings(embeddings, num_speakers=num_speakers)
        
        return speaker_lables
    
    def estimate_num_speakers(self, X: np.ndarray, max_speakers: int = 10) -> int:
        from sklearn.metrics import silhouette_score

        n_samples = len(X)
        if n_samples < 2:
            # Not enough data to estimate number of speakers
            return 1

        max_possible_clusters = min(max_speakers, n_samples - 1)
        if max_possible_clusters < 2:
            return 1  # Not enough clusters to estimate

        scores: List[Tuple[int, float]] = []

        for n_clusters in range(2, max_possible_clusters + 1):
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average"
            )
            labels: np.ndarray = clustering.fit_predict(X)
            # Check if the number of unique labels is valid
            num_labels = len(set(labels))
            if num_labels >= 2 and num_labels < n_samples:
                score: float = silhouette_score(X, labels, metric="cosine")
                scores.append((n_clusters, score))
            else:
                continue  # Skip invalid clustering results

        if scores:
            # Find the n_clusters with the highest silhouette score
            best_n_clusters = max(scores, key=lambda x: x[1])[0]
            return best_n_clusters
        else:
            return 1  # Default to one speaker if not enough data
