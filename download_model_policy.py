from google.cloud import storage
INPUT_CNN_PATH = "input_cnn.pt"
#POLICY_INPUT_CNN_PATH = "policy_input_cnn.pt"
MODEL_PATH = "recurrent_model.pt"
POLICY_PATH = "my_policy.pt"
client = storage.Client()
bucket = client.get_bucket("mpc-test")
blob = bucket.blob(INPUT_CNN_PATH)
blob.download_to_filename(INPUT_CNN_PATH)
blob = bucket.blob(MODEL_PATH)
blob.download_to_filename(MODEL_PATH)
blob = bucket.blob(POLICY_PATH)
blob.download_to_filename(POLICY_PATH)
#blob = bucket.blob(POLICY_INPUT_CNN_PATH)
#blob.download_to_filename(POLICY_INPUT_CNN_PATH)
