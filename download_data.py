from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset

for dataset_name in ['traffic_nips', 'solar_nips', 'electricity_nips', 'taxi_30min', 'exchange_rate']:
    dataset = get_dataset(dataset_name, regenerate=False)
