import pandas as pd
from sklearn.neighbors import NearestNeighbors


class Pipeline:
    def __init__(self) -> None:
        pass

    @staticmethod
    def rename_columns(ratings, books, users):
        ratings.columns = ["userID", "ISBN", "bookRating"]
        books.columns = ["ISBN", "bookTitle",
                         "bookAuthor", "yearOfPublication", "publisher",
                         "ImageURLS", "ImageURLM", "ImageURLL"]
        users.columns = ["userID", "Location", "Age"]
        return ratings, books, users

    @staticmethod
    def remove_img_urls(dataset):
        dataset = dataset.drop(
            ["ImageURLS", "ImageURLM", "ImageURLL"], axis=1)
        return dataset

    @staticmethod
    def find_k_nearest_neighbors(user_index, csr_matrix, categories, nn_model):
        _, nearest_neighbors = nn_model.kneighbors(csr_matrix[user_index])

        return pd.Categorical.from_codes(nearest_neighbors[0],
                                         categories=categories)

    @staticmethod
    def knn_age_imputing(row, csr, categories, df, mapping, nn_model):
        if pd.isna(row.Age):
            #find nearest neighbors according to the books he rates
            neigbours = Pipeline.find_k_nearest_neighbors(mapping.get(row.userID), csr,
            categories = categories, nn_model=nn_model)

            #remove NAs and compute mean age of these neighbors
            try:
                return df.loc[df.userID.isin(neigbours)].dropna().Age.mean()
            except:
                #returns overall mean if no neigbors have age
                return 200000
        else:
            return row.Age
