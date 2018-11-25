from google.cloud import storage
MODEL_PATH = "my_model.pt"
POLICY_PATH = "my_policy.pt"
client = storage.Client()
bucket = client.get_bucket("mpc-test")
blob = bucket.blob(MODEL_PATH)
blob.download_to_filename(MODEL_PATH)
blob = bucket.blob(POLICY_PATH)
blob.download_to_filename(POLICY_PATH)
