import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy
import json
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, normalize


class Pipeline:
    MAPPING_PATH = "./../data/similarity_mapping.json"

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
            # find nearest neighbors according to the books he rates
            neigbours = Pipeline.find_k_nearest_neighbors(mapping.get(row.userID), csr,
                                                          categories=categories, nn_model=nn_model)
            # remove NAs and compute mean age of these neighbors
            return df.loc[df.userID.isin(neigbours)].dropna().Age.mean()
        else:
            return row.Age

    @staticmethod
    def create_item_item_similarity_mapping():
        combined_data = pd.read_csv("./../data/merged_data.csv")
        # deleting implicit feedback
        combined_data.loc[combined_data.bookRating == 0,["bookRating"]] = 5
        # Reducing number of users
        users_high = combined_data.groupby("userID").count()["bookRating"] > 100
        ids_high = users_high[users_high].index.tolist()
        combined_data = combined_data.loc[combined_data.userID.isin(ids_high)].copy()
        # Reducing number of books
        books_high = combined_data.groupby("bookTitle").count()["bookRating"] > 50
        books_ids_high = books_high[books_high].index.tolist()
        combined_data = combined_data.loc[combined_data.bookTitle.isin(
            books_ids_high)].copy()
        combined_data.userID = combined_data.userID.astype("category")
        combined_data.bookTitle = combined_data.bookTitle.astype("category")
        users = combined_data["userID"].unique()
        book_titles = combined_data["bookTitle"].unique()
        shape = (len(book_titles), len(users))
        # setting implicit feedback to 3 for now
        print(len(combined_data))
        print(shape)
        coo = scipy.sparse.coo_matrix((combined_data.bookRating,
                                       (combined_data.bookTitle.cat.codes,
                                        combined_data.userID.cat.codes)),
                                      shape=shape)
        csr = coo.tocsr()
        # normalizing user ratings (features)
        csr = normalize(csr, norm="l1", axis=0)
        # Appending book biblio data to csr matrix
        #columns = ["bookAuthor", "publisher"]
        #biblio_data = []
        #combined_data.drop_duplicates(subset=["bookTitle"], inplace=True)
        #for column in columns:
        #    one_hot = OneHotEncoder()
        #    biblio_data.append(one_hot.fit_transform(combined_data.sort_values(by="bookTitle")[[column]]))
        # ordinally encoding years
        #ordinal = OrdinalEncoder()
        #biblio_data.append(ordinal.fit_transform(
        #    combined_data.sort_values(by="bookTitle")[["yearOfPublication"]]))
        #csr = scipy.sparse.hstack((csr, biblio_data[0], biblio_data[1],
        #                           biblio_data[2]))
        print(csr.shape)
        nn_model = NearestNeighbors(n_neighbors=6, algorithm="brute",
                                    metric="cosine", n_jobs=-1)
        nn_model.fit(csr)
        _, nearest_neighbors = nn_model.kneighbors(csr)
        cat_nearest_books = pd.Categorical.from_codes(nearest_neighbors,
                                                      categories=combined_data.bookTitle.cat.categories)
        mapping = {neigbours[0]: list(neigbours[1:]) for neigbours in cat_nearest_books}
        with open(Pipeline.MAPPING_PATH, "w") as json_file:
            json.dump(mapping, json_file)

    @staticmethod
    def get_item_item_similar_mapping() -> list:
        with open(Pipeline.MAPPING_PATH, "r") as json_file:
            item_items_mapping = json.load(json_file)
        return item_items_mapping
