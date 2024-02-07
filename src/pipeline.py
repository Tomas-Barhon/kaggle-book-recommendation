import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy

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
            return df.loc[df.userID.isin(neigbours)].dropna().Age.mean()
        else:
            return row.Age

    @staticmethod
    def create_item_item_similarity_mapping():
        combined_data = pd.read_csv("./../data/merged_data.csv")
        combined_data.userID = combined_data.userID.astype("category")
        combined_data.bookTitle = combined_data.bookTitle.astype("category")
        users = combined_data["userID"].unique()
        book_titles = combined_data["bookTitle"].unique()
        shape = (len(book_titles), len(users))
        coo = scipy.sparse.coo_matrix((combined_data.bookRating,
                        (combined_data.bookTitle.cat.codes,
                         combined_data.userID.cat.codes)),
                                    shape=shape)
        csr = coo.tocsr()
        nn_model = NearestNeighbors(n_neighbors=5, algorithm="auto",
                            metric="cosine")
        nn_model.fit(csr)
        _, nearest_neighbors = nn_model.kneighbors(csr)
        cat_nearest_books = pd.Categorical.from_codes(nearest_neighbors,
                                         categories=combined_data.bookTitle.cat.categories)
        return {title:list(similar.to_numpy()) for title, similar in zip(combined_data.bookTitle.unique(),cat_nearest_books)}
    @staticmethod
    def get_item_item_similar(combined_data) -> list:
        users = combined_data["userID"].unique()
        book_titles = combined_data["bookTitle"].unique()
        shape = (len(users), len(book_titles))
        coo = scipy.sparse.coo_matrix((combined_data.bookRating,
                        (combined_data.userID.cat.codes,
                    combined_data.bookTitle.cat.codes)),
                                    shape=shape)
        csr = coo.tocsr()
        