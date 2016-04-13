from openml.apiconnector import APIConnector


apikey = "1cfa29e70221198faaa83b0aaa97c60c"
connector = APIConnector(apikey=apikey)
dataset_id = 44
dataset = connector.download_dataset(dataset_id)

# 4601 samples 57 features
def return_data():
	X, y, attribute_names = dataset.get_dataset(target=dataset.default_target_attribute, return_attribute_names=True)
	return X, y, attribute_names