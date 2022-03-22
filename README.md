# phi_dataset_preprocessing

import sys </br>
!test -d phi_dataset_preprocessing || git clone https://github.com/cappelchi/phi_dataset_preprocessing.git </br>
if not 'phi_dataset_preprocessing' in sys.path: </br>
  sys.path += ['phi_dataset_preprocessing'] </br>
!ls phi_dataset_preprocessing </br>
 </br>
from phi_dataset_preprocessing.dataset import football </br>
#upload yaml config </br>
matches = football('./base_line_data.yaml') </br>
live_df = matches.download_parse_clean_normalize() </br>
 </br>
#options: </br>
#force_parsing = False(non parse results if exist) </br>
#dataset4 = 'training' #('validation') </br>
#system4 = 'windows' #csv line termnator windows/linux ('\r\n', '\n') </br>
#draw_distribution = False #show distribution via plotly </br>
#pdf_report = False #make pdf report with distributions </br>
