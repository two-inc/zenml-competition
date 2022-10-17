import pandas

BUCKET_URL = "http://storage.googleapis.com/zenmldata/"
DEFAULT_OBJECT_NAME = "bs140513_032310.csv"


def load_data(object_name: str = None):
    object_url = BUCKET_URL + (object_name or DEFAULT_OBJECT_NAME)
    return pandas.read_csv(object_url)
