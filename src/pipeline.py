import pandas as pd


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


