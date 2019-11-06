import numpy as np

# folders
zone_folder                     = 'zone'
output_data_folder              = 'data'
dataset_path                    = 'dataset'
threshold_map_folder            = 'threshold_map'
models_information_folder       = 'models_info'
results_information_folder      = 'results'
saved_models_folder             = 'saved_models'
min_max_custom_folder           = 'custom_norm'
learned_zones_folder            = 'learned_zones'

# files or extensions
csv_model_comparisons_filename  = 'models_comparisons.csv'
seuil_expe_filename             = 'seuilExpe'
min_max_filename_extension      = '_min_max_values'

# variables 
renderer_choices                = ['all', 'maxwell', 'igloo', 'cycle']

scenes_names                    = ['Appart1opt02', 'Bureau1', 'Cendrier', 'Cuisine01', 'EchecsBas', 'PNDVuePlongeante', 'SdbCentre', 'SdbDroite', 'Selles']
scenes_indices                  = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

maxwell_scenes_names            = ['Appart1opt02', 'Cuisine01', 'SdbCentre', 'SdbDroite']
maxwell_scenes_indices          = ['A', 'D', 'G', 'H']

igloo_scenes_names              = ['Bureau1', 'PNDVuePlongeante']
igloo_scenes_indices            = ['B', 'F']

cycle_scenes_names              = ['EchecBas', 'Selles']
cycle_scenes_indices            = ['E', 'I']

zones_indices                   = np.arange(16)

# parameters
scene_image_quality_separator     = '_'
scene_image_extension             = '.png'