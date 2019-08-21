####################

from Code.image_preprocessing.T1_preprocessing import preprocessing_t1w

## run the pipeline
#  for test
##
bids_directory = '/network/lustre/dtlake01/aramis/users/clinica/CLINICA_datasets/BIDS/ADNI_BIDS_T1_new'
caps_directory= '/network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/Results/CAPS'
tsv= '/network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/tsv_files/image_preprocessing_test.tsv'
working_dir = '/network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/Results/working_dir'
ref_template = '/network/lustre/dtlake01/aramis/users/junhao.wen/from_gpfs/PhD/ADNI_classification/gitlabs/AD-DL/Data/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii'


wf = preprocessing_t1w(bids_directory, caps_directory, tsv, ref_template, working_directory=working_dir)
wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})
