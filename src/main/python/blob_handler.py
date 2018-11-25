from google.cloud import storage


class BlobHandler:
    def __init__(self, bucket_name):
        client = storage.Client()
        self.bucket = client.get_bucket(bucket_name)

    def ls_blob(self):
        blob_name_list = [blob.name for blob in self.bucket.list_blobs()]
        return blob_name_list

    def upload_blob(self, filename):
        blob = self.bucket.blob(filename)
        blob.upload_from_filename(filename)

    def download_blob(self, filename):
        blob = self.bucket.blob(filename)
        blob.download_to_filename(filename)
