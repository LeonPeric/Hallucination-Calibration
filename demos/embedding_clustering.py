import sys
import numpy as np
import sys
import pandas as pd
import torch
import json
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

def save_dist_classification_results(run, accuracy, precision, recall,f1, confusion_matrix, save_path):
    experiment = {}
    experiment['run'] = run
    experiment['accuracy'] = accuracy
    experiment['precision'] = precision
    experiment['recall'] = recall
    experiment['f1'] = f1
    experiment['confusion_matrix'] = confusion_matrix.tolist()

    json_str = json.dumps(experiment)
    with open(save_path, 'a') as file:
        file.write(json_str + '\n')

def load_embeddings(embeddings_path):
    loaded_embeddings = []
    embeddings = []

    with open(embeddings_path, "r") as f:
        for line in f:
            loaded_embeddings.append(json.loads(line))
    for loaded_fact, loaded_embedding in loaded_embeddings:
        embeddings.append((loaded_fact, torch.tensor(loaded_embedding)))

    return embeddings



def get_clustering_stats(all_training_embeddings_numpy,hallucinations_embeddings_numpy,generated_true_facts_embeddings_numpy,monofact_len,normal_fact_len, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_training_embeddings_numpy)

    training_labels = kmeans.labels_
    if len(hallucinations_embeddings_numpy) > 0:
        hallucination_labels = kmeans.predict(hallucinations_embeddings_numpy)
    true_facts_labels = kmeans.predict(generated_true_facts_embeddings_numpy)
    if len(hallucinations_embeddings_numpy) > 0:
        all_labels = np.concatenate([training_labels, hallucination_labels, true_facts_labels])
    else:
        all_labels = np.concatenate([training_labels, true_facts_labels])

    clustered_points = pd.DataFrame({'cluster':all_labels})


    clustered_points['isMonofact'] = (clustered_points.index < monofact_len).astype(int)
    monofact_rate = clustered_points["isMonofact"].mean()

    clustered_points["isNormal_fact"] = ((monofact_len <= clustered_points.index) &
                                            (clustered_points.index < monofact_len + normal_fact_len)).astype(int)

    clustered_points['isHallucination'] = ((monofact_len + normal_fact_len <= clustered_points.index) &
                                            (clustered_points.index < monofact_len + normal_fact_len +
                                             len(hallucinations_embeddings_numpy))).astype(int)

    hallucination_rate = len(hallucinations_embeddings_numpy)/len(generated_true_facts_embeddings_numpy)

    clustered_points['isGenerated_True_fact'] = ((monofact_len + len(hallucinations_embeddings_numpy) + normal_fact_len )
                                            <= clustered_points.index).astype(int)



    grouped_cluster = clustered_points.groupby('cluster').sum()
    grouped_cluster["cluster_hallucination_rate"] = (grouped_cluster["isHallucination"]/
                                                     (grouped_cluster["isGenerated_True_fact"] + grouped_cluster["isHallucination"]))
    grouped_cluster["cluster_monofact_rate"] = (grouped_cluster["isMonofact"] /
                                                     (grouped_cluster["isMonofact"] + grouped_cluster[
                                                         "isNormal_fact"]))
    grouped_cluster["above_mean_hallucination_rate"] = (grouped_cluster["cluster_hallucination_rate"] > hallucination_rate).astype(int)
    grouped_cluster["above_mean_monofact_rate"] = (
                grouped_cluster["cluster_monofact_rate"] > monofact_rate).astype(int)
    grouped_cluster["hallucination_monofact_correlation"] = (grouped_cluster["above_mean_hallucination_rate"] ==
                                                             grouped_cluster["above_mean_monofact_rate"]).astype(int)
    return grouped_cluster["hallucination_monofact_correlation"].mean()

def l1_distances(list1, list2):

    if len(list1) == 0:
        return []
    tensors1 = torch.stack(list1)
    tensors2 = torch.stack(list2)

    # Reshape tensors for broadcasting
    tensor1 = tensors1.unsqueeze(1)  # shape (n, 1, d)
    tensor2 = tensors2.unsqueeze(0)  # shape (1, m, d)

    # Compute L1 distances
    distances = torch.abs(tensor1 - tensor2).sum(dim=2)
    closest_index = torch.argmin(distances, dim=1)
    return closest_index.tolist()

runs = 60

for run in range(runs):
    if run < 16:
        run += 1
        continue

    generated_fact_embeddings = load_embeddings(f"../src/experiment/runs/embeddings/Run_{run}/embeddings/generated_fact_embeddings.json")
    hallucination_embeddings = load_embeddings(f"../src/experiment/runs/embeddings/Run_{run}/embeddings/hallucination_embeddings.json")
    monofact_embeddings = load_embeddings(f"../src/experiment/runs/embeddings/Run_{run}/embeddings/monofact_embeddings.json")
    normal_fact_embeddings = load_embeddings(f"../src/experiment/runs/embeddings/Run_{run}/embeddings/normal_fact_embeddings.json")

    all_training_embeddings = []

    for _, embed in monofact_embeddings:
        all_training_embeddings.append(embed)

    for _, embed in normal_fact_embeddings:
        all_training_embeddings.append(embed)

    all_training_embeddings_numpy = torch.cat([a.unsqueeze(0) for a in all_training_embeddings]).detach().numpy()


    monofacts = [fact for fact,_ in monofact_embeddings]


    hallucinations = [fact for fact, _ in hallucination_embeddings]


    hallucinations_embeddings_numpy =  np.array([a.detach().numpy() for _,a in hallucination_embeddings])



    normal_facts = [fact for fact, _ in normal_fact_embeddings]


    generated_fact_embeds = []

    generated_true_fact_embeds = []
    for fact, embed in generated_fact_embeddings:
        generated_fact_embeds.append(embed)
        if fact in hallucinations:
            continue
        else:
            generated_true_fact_embeds.append(embed)

    generated_true_facts_embeddings_numpy = np.array([a.detach().numpy() for a in generated_true_fact_embeds])

    generated_fact_closest_embed_index  = l1_distances(generated_fact_embeds, all_training_embeddings)

    preds = []
    for idx in generated_fact_closest_embed_index:
        if idx < len(monofact_embeddings):
            preds.append(1)
        else:
            preds.append(0)


    labels = []
    for fact, embed in generated_fact_embeddings:
        if fact in hallucinations:
            labels.append(1)
        else:
            labels.append(0)

    conf_matrix = confusion_matrix(preds, labels, labels=[0, 1])


    accuracy = accuracy_score(preds, labels)
    precision = precision_score(preds, labels, average='weighted')
    recall = recall_score(preds, labels, average='weighted')
    f1 = f1_score(preds, labels, average='weighted')

    classification_results_path = "../src/experiment/results/classification_results.json"
    save_dist_classification_results(run=run,accuracy=accuracy,precision=precision,recall=recall,
                                     f1=f1, confusion_matrix=conf_matrix,save_path=classification_results_path)
    cluster_results = {}
    cluster_results["run"] = run
    cluster_results["3_clusters"] = get_clustering_stats(all_training_embeddings_numpy,hallucinations_embeddings_numpy,generated_true_facts_embeddings_numpy,len(monofacts),len(normal_facts), n_clusters=3)
    cluster_results["4_clusters"] = get_clustering_stats(all_training_embeddings_numpy,hallucinations_embeddings_numpy,generated_true_facts_embeddings_numpy,len(monofacts),len(normal_facts), n_clusters=4)
    cluster_results["5_clusters"] = get_clustering_stats(all_training_embeddings_numpy,hallucinations_embeddings_numpy,generated_true_facts_embeddings_numpy,len(monofacts),len(normal_facts), n_clusters=5)
    cluster_results["6_clusters"] = get_clustering_stats(all_training_embeddings_numpy,hallucinations_embeddings_numpy,generated_true_facts_embeddings_numpy,len(monofacts),len(normal_facts), n_clusters=6)
    cluster_results["7_clusters"] = get_clustering_stats(all_training_embeddings_numpy,hallucinations_embeddings_numpy,generated_true_facts_embeddings_numpy,len(monofacts),len(normal_facts), n_clusters=7)

    json_str = json.dumps(cluster_results)
    with open("../src/experiment/results/cluster_results.json", 'a') as file:
        file.write(json_str + '\n')











