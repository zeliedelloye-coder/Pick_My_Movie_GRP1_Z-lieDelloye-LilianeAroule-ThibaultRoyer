# Pick_My_Movie_GRP1_ZelieDelloye-LilianeAroule-ThibaultRoyer
Un projet académique de Machine learning à l'ESME du groupe 1 : Zélie Delloye, Liliane Aroule et Thibault Royer.
La première étape pour tous les types fichiers est de les télécharger. Pensez à bien mettre "PickMyMovie-GRP1" dans le fichier "datas" et le fichier "PickMyMovie Rapport final GRP1" dans le fichier "images". ATTENTION VÉRIFIER QUE PANDA EST À JOUR !
IL EST TRÈS IMPORTANT D'ETRE SUR UN ENVIRONNEMENT CONDA ENV:ML
Pour le fichier "PickMyMovie-GRP1", veuillez à bien avoir dans le même dossier les fichier csv suivant :
- genome-scores.csv
- genome-tags.csv
- links.csv
- movies.csv
Deux solutions pour lancer le projet : lancer toutes les tuiles du notebook les unes après les autres SAUF la dernière OU BIEN lancer que la dernière tuile qui fonctionne avec l'interface.
Il est à noter que quelque soit le votre choix, l'étape "model = api.load("fasttext-wiki-news-subwords-300")" (Toute première étape sur l'interface) est une étape qui va prendre plusieurs importantes minutes car le model est d'environ 1 Go.
Si vous avez un problème avec la 3ème case, Utiliser celle-ci
import pandas as pd

movies = pd.read_csv("movies.csv", engine="python")
ratings = pd.read_csv("ratings.csv", nrows=200000, engine="python")
genome_scores = pd.read_csv("genome-scores.csv", engine="python")
genome_tags = pd.read_csv("genome-tags.csv", engine="python")

movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")

genome_scores["movieId"] = pd.to_numeric(genome_scores["movieId"], errors="coerce")
genome_scores["tagId"] = pd.to_numeric(genome_scores["tagId"], errors="coerce")

genome_tags["tagId"] = pd.to_numeric(genome_tags["tagId"], errors="coerce")

# on supprime les lignes cassées
movies = movies.dropna(subset=["movieId"])
ratings = ratings.dropna(subset=["movieId"])
genome_scores = genome_scores.dropna(subset=["movieId", "tagId"])
genome_tags = genome_tags.dropna(subset=["tagId"])

# conversion finale propre
movies["movieId"] = movies["movieId"].astype(int)
ratings["movieId"] = ratings["movieId"].astype(int)
genome_scores["movieId"] = genome_scores["movieId"].astype(int)
genome_scores["tagId"] = genome_scores["tagId"].astype(int)
genome_tags["tagId"] = genome_tags["tagId"].astype(int)

movies["year"] = pd.to_numeric(
    movies["title"].str.extract(r"\((\d{4})\)")[0],
    errors="coerce"
)

movies["title_clean"] = (
    movies["title"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(r"\(.*?\)", "", regex=True)
    .str.strip()
)

ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")

df_merged = ratings.merge(movies, on="movieId", how="inner")

movie_stats = df_merged.groupby("title_clean").agg(
    mean_rating=("rating", "mean"),
    num_ratings=("rating", "count"),
    year=("year", "first"),
    genres=("genres", "first")
)

movie_counts = ratings["movieId"].value_counts()
popular_movies = movie_counts[movie_counts >= 50].index

ratings_small = ratings[ratings["movieId"].isin(popular_movies)]

movie_vectors = genome_scores.pivot_table(
    index="movieId",
    columns="tagId",
    values="relevance",
    fill_value=0
)

movie_vectors = movie_vectors.loc[
    movie_vectors.index.intersection(ratings_small["movieId"])
]

genome = (
    genome_scores
    .merge(genome_tags, on="tagId", how="left")
    .merge(movies[["movieId", "title"]], on="movieId", how="left")
)

Pour le fichier "PickMyMovie Rapport final GRP1" veuillez bien téléchargez toutes les images et les mettre dans le même fichier que le notebook.
NE PAS LANCER LES CASES AVEC DU CODE, mais pour afficher les images, il faut exécuter les tuiles avec un code d'image comme celui ci "<img src="images/similarités_matrix.png" width="500">". Sinon il ne faut pas éxecuter.



