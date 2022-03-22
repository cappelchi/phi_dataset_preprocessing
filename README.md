# phi_dataset_preprocessing

import sys

!test -d phi_dataset_preprocessing || git clone https://github.com/cappelchi/phi_dataset_preprocessing.git
if not 'phi_dataset_preprocessing' in sys.path:
  sys.path += ['phi_dataset_preprocessing']
!ls phi_dataset_preprocessing

from phi_dataset_preprocessing.dataset import football
#upload yaml config
matches = football('./base_line_data.yaml')
live_df = matches.download_parse_clean_normalize()

#options:
#force_parsing = False(non parse results if exist)
#dataset4 = 'training' #('validation')
#system4 = 'windows' #csv line termnator windows/linux ('\r\n', '\n')
#draw_distribution = False #show distribution via plotly
#pdf_report = False #make pdf report with distributions
