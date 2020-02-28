#!~/.guix-profile/bin/python3

import sys
import glob
import gzip
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import pysam
import numpy as np

"""
Script by: Warren Winick

This script takes two input segregation tables, and will generate for each chromosome:
-NPMI matrices 
-Z-Score matrices
-Differential Z-Score matrices
-Top 5% significant differential matrices
-Strongest 10% common contacts
For each of the above, it is possible to produce full length and/or distance cut-off matrices (default 10Mb)

In addition, the script will generate:
-DIfferential window counts from top 5% significant contact matrices
-Top 10% differential windows for each dataset

This script also determines lowly detected windows (default 2%) in each input segregation table and removes them from both datasets

"""



# get data

def open_segregation(path_or_buffer):
	"""
	Open a segregation table from a file.
	:param path_or_buffer: Path to imput segregation table, or open python file object.
	:returns: :ref:`segregation table <segregation_table>`
	"""
	open_seg = pd.read_csv(path_or_buffer, sep='\t') #, nrows=100
	open_seg.set_index(['chrom','start','stop'], inplace = True)	
	return open_seg

def calculate_NPMI(seg_matrix):
	np.warnings.filterwarnings('ignore')	
	# calculate M the number of samples
	M = len(seg_matrix[0])

	#calculate p(x,y) matrix i.e coseg matrix divided my M
	pxy = (seg_matrix.dot(seg_matrix.T))/M

	#p(x)p(y) matrix  - i.e detection frequency with the dot product of it's transposed self equals an N*N matrix
	pxpy = seg_matrix.sum(1).reshape(-1,1)/M * seg_matrix.sum(1)/M

	#define PMI matrix
	PMI = np.log2(pxy/pxpy)

	#bound values between -1 and 1
	NPMI = PMI/-np.log2(pxy)

	return NPMI



# filter suspicious regions

def windows_detection_frequencies (segregation_table):
	import pandas as pd
	import numpy as np
	if len (segregation_table.index.names) == 1: # Check if supplied segregation table has multiindex
		segregation_table.set_index(['chrom','start','stop'], inplace = True)
	number_of_samples = len(list(segregation_table))
	frequencies_list = []
	for index, row in segregation_table.iterrows(): # Itterate over rows of the segregation tables
		number_of_positive = (row == 1).values.sum()
		frequency = float(number_of_positive/number_of_samples)
		frequencies_list.append(frequency)
	frequencies_list = np.around(np.array(frequencies_list),4) # Keep first 4 digits after decimal point
	frequencies_df = pd.DataFrame(index=segregation_table.index)
	frequencies_df['data'] = pd.DataFrame(frequencies_list, index=segregation_table.index)

	return frequencies_df

def get_bad_index_list_from_WDF(frequencies_df1, frequencies_df2):
	'''
	Generate a list of windows which need to be masked in the Z-score calculation.
	:param frequencies_df1, frequencies_df2: dataframe of window detection frequencies with column 'data' 
	'''
	
	#generate df for each frequency list that has zeros in place of lowest 2% of frequencies
	noZerodf1 = frequencies_df1.loc[~(frequencies_df1==0).all(axis=1)]
	greater_than_2per_array1 = np.where(frequencies_df1['data'] < (np.nanpercentile(noZerodf1, 2)), 0, frequencies_df1['data']) #determine lowest 2% of windows from dataframe1
	greater_than_2per_array_df1 = pd.DataFrame(greater_than_2per_array1, index=frequencies_df1.index)
	greater_than_2per_array_df1.columns = ['data']
	noZerodf2 = frequencies_df2.loc[~(frequencies_df2==0).all(axis=1)]
	greater_than_2per_array2 = np.where(frequencies_df2['data'] < (np.nanpercentile(noZerodf2, 2)), 0, frequencies_df2['data']) #determine lowest 2% of windows from dataframe2
	greater_than_2per_array_df2 = pd.DataFrame(greater_than_2per_array2, index=frequencies_df2.index)
	greater_than_2per_array_df2.columns = ['data']
	
	#generate list of zero windows for each dataset
	bad_index_zeros1 = greater_than_2per_array_df1.loc[greater_than_2per_array_df1['data'] == 0]
	bad_index_zeros2 = greater_than_2per_array_df2.loc[greater_than_2per_array_df2['data'] == 0]
	bad_index_zeros1_reset = bad_index_zeros1.reset_index()
	bad_index_zeros2_reset = bad_index_zeros2.reset_index()
	bad_index_zeros1_reset['index_list'] = (bad_index_zeros1_reset['chrom'] + ':' + bad_index_zeros1_reset['start'].map(str) + '-' + bad_index_zeros1_reset['stop'].map(str)).astype(str)
	bad_index_zeros2_reset['index_list'] = (bad_index_zeros2_reset['chrom'] + ':' + bad_index_zeros2_reset['start'].map(str) + '-' + bad_index_zeros2_reset['stop'].map(str)).astype(str)
	zeros_index_list1 = bad_index_zeros1_reset['index_list'].values.tolist()
	zeros_index_list2 = bad_index_zeros2_reset['index_list'].values.tolist()
	
	#determine non-overlapping <2% windows in both datasets and create a list with only with these windows
	greater_than_2per_array_df1['match'] = np.where((greater_than_2per_array_df2['data'] == 0) == (greater_than_2per_array_df1['data'] > 0), 'Yes', 'No') 
	greater_than_2per_array_df2['match'] = np.where((greater_than_2per_array_df1['data'] == 0) == (greater_than_2per_array_df2['data'] > 0), 'Yes', 'No') 
	bad_sample_df1_noNA = (greater_than_2per_array_df1.where(greater_than_2per_array_df1['match'].isin(['Yes']))).dropna()
	bad_sample_df2_noNA = (greater_than_2per_array_df2.where(greater_than_2per_array_df2['match'].isin(['Yes']))).dropna()
	bad_sample_df1_noNA_reset = bad_sample_df1_noNA.reset_index()
	bad_sample_df2_noNA_reset = bad_sample_df2_noNA.reset_index()
	bad_sample_df1_noNA_reset['index_list'] = (bad_sample_df1_noNA_reset['chrom'] + ':' + bad_sample_df1_noNA_reset['start'].map(str) + '-' + bad_sample_df1_noNA_reset['stop'].map(str)).astype(str)
	bad_sample_df2_noNA_reset['index_list'] = (bad_sample_df2_noNA_reset['chrom'] + ':' + bad_sample_df2_noNA_reset['start'].map(str) + '-' + bad_sample_df2_noNA_reset['stop'].map(str)).astype(str)
	bad_index_list1 = bad_sample_df1_noNA_reset['index_list'].values.tolist()
	bad_index_list2 = bad_sample_df2_noNA_reset['index_list'].values.tolist()
	
	#create a list of windows to be masked for each df
	final_indexlist1 = bad_index_list1 + zeros_index_list1
	final_indexlist2 = bad_index_list2 + zeros_index_list2
	
	return final_indexlist1, final_indexlist2


# normalize matrix
	
def zscore_matrix (NPMI_matrix):
	"""
	Calculate Z-score for each diagonal of an input matrix
	:returns: normalized z-score matrix with index and column names from orginal file
	"""
	chrom_matrix = np.ma.masked_invalid(NPMI_matrix)
	z_matrix = np.zeros_like(chrom_matrix)
	for i in range(1, len(chrom_matrix)-1):
		diag = np.diagonal(chrom_matrix, offset=i) 
		zdiag = stats.zscore(diag) 
		np.fill_diagonal(z_matrix[:,i:], zdiag)
	Z_matrix = z_matrix + z_matrix.T
	return Z_matrix


# extract differential contacts

def get_top5_bottom5_matrices (diff_matrix):
	"""
	Calculate top 5% of contacts for each dataset in a differential Z-Score matrix
	"""
	flat_matrix = np.matrix.flatten(diff_matrix)
	data_flat_matrix = flat_matrix[~np.isnan(flat_matrix)]
	(mu_diff_matrix, sigma_diff_matrix) = norm.fit(data_flat_matrix)
	top5_matrix2 = norm.ppf(0.05, loc=mu_diff_matrix, scale=sigma_diff_matrix)
	top5_matrix1 = norm.ppf(0.95, loc=mu_diff_matrix, scale=sigma_diff_matrix)
	matrix1 = diff_matrix.copy()
	matrix2 = diff_matrix.copy()
	matrix1[matrix1 <= top5_matrix1] = np.nan
	matrix2[matrix2 > top5_matrix2] = np.nan
	return matrix1, matrix2

def get_common_top10_matrix (diff_matrix, Zmatrix1, Zmatrix2):
	""""""
	#get value for +/- 1 standard deviation from mean	
	flat_matrix = np.matrix.flatten(diff_matrix)
	data_flat_matrix = flat_matrix[~np.isnan(flat_matrix)]
	(mu_diff_matrix, sigma_diff_matrix) = norm.fit(data_flat_matrix)
	common_matrix1_score = mu_diff_matrix+sigma_diff_matrix
	common_matrix2_score = mu_diff_matrix-sigma_diff_matrix

	#mask all values outside +/- 1 std. dev. in differential matrix
	max_and_min_matrix = diff_matrix.copy()
	common_matrix = np.ma.masked_where((common_matrix2_score > max_and_min_matrix), max_and_min_matrix)
	common_matrix2 = np.ma.masked_where((common_matrix > common_matrix1_score), common_matrix)
	common_matrix2_filled = np.ma.filled(common_matrix2, fill_value=np.nan)
	common_matrix_formasking = np.ma.masked_invalid(common_matrix2_filled)

	#apply diff matrix mask to original Z-Score matrices and fill mask with NaN 	
	masked_Zmatrix1 = np.ma.masked_where((common_matrix_formasking == True), Zmatrix1)
	masked_Zmatrix2 = np.ma.masked_where((common_matrix_formasking == True), Zmatrix2)
	masked_Zmatrix1_filled = np.ma.filled(masked_Zmatrix1, fill_value=np.nan)
	masked_Zmatrix2_filled = np.ma.filled(masked_Zmatrix2, fill_value=np.nan)

	#get minimum Z-Score value for each window
	minimum_matrix = np.minimum(masked_Zmatrix1_filled, masked_Zmatrix2_filled)
	minimum_matrix_flatten = np.matrix.flatten(minimum_matrix)
	minimum_matrix_flatten_noNA = minimum_matrix_flatten[~np.isnan(minimum_matrix_flatten)]
	values_for_list = minimum_matrix_flatten_noNA.tolist()

	#sort values in list and obtain the 10% threshold value
	compare_R1_R2 = pd.DataFrame(values_for_list, columns=['values'], index=None)
	compare_R1_R2_sorted = compare_R1_R2.sort_values(by='values')
	compare_R1_R2_sorted_top10 = compare_R1_R2_sorted.tail(int(len(compare_R1_R2_sorted)*0.1))
	compare_R1_R2_sorted_top10.reset_index(drop=True)
	threshold = compare_R1_R2_sorted_top10['values'].iloc[0]

	#mask minimum matrix below threshold and apply to differential common matrix
	minimum_matrix_masked = np.ma.masked_invalid(minimum_matrix)
	masking_below_threshold = np.ma.masked_where((minimum_matrix_masked < threshold), minimum_matrix_masked)
	masked_common_matrix2 = np.ma.masked_where((masking_below_threshold == True), common_matrix2)
	masked_common_matrix2_fillNA = np.ma.filled(masked_common_matrix2, fill_value=np.nan)
	return masked_common_matrix2_fillNA

def get_differential_Z_score_windowscore(zchrom1, zchrom2, chrom, segregation_chrom):
	chrom_sizes = {"chr1" : 195471971,"chr2" : 182113224,"chr3" : 160039680,"chr4" : 156508116, \
            "chr5" : 151834684,"chr6" : 149736546,"chr7" : 145441459,"chr8" : 129401213, \
            "chr9" : 124595110,"chr10" : 130694993, "chr11" : 122082543,"chr12" : 120129022, \
            "chr13" : 120421639,"chr14" : 124902244,"chr15" : 104043685,"chr16" : 98207768, \
            "chr17" : 94987271,"chr18" : 90702639,"chr19" : 61431566,"chrX": 171031299,"chrY" : 91744698}  	
	value_list = []
	end_region = int(chrom_sizes[chrom])
	col1_name = segregation_chrom.columns[0]
	col2_name = segregation_chrom.columns[1]
	coordinate1 = col1_name.split(':')[1]
	coordinate2 = col2_name.split(':')[1]
	start1 = int(coordinate1.split('-')[0])
	start2 = int(coordinate2.split('-')[0])
	interval_size = int(start2 - start1)
	sliding_start = 0
	sliding_stop = int(sliding_start+interval_size)
	while sliding_stop < end_region:
		start_bin = int(sliding_start/interval_size)
		column_value1 = zchrom1.iloc[start_bin].fillna(0).astype(bool).sum(axis=0)
		column_value2 = zchrom2.iloc[start_bin].fillna(0).astype(bool).sum(axis=0)
		value_for_list = column_value1 - column_value2
		value_list.append(value_for_list)
		sliding_start = int(sliding_start+interval_size)
		sliding_stop = int(sliding_stop+interval_size)
	column_end1 = zchrom1.iloc[-1].fillna(0).astype(bool).sum(axis=0)
	column_end2 = zchrom2.iloc[-1].fillna(0).astype(bool).sum(axis=0)
	end_value_for_list = column_end1 - column_end2
	value_list.append(end_value_for_list)
	return value_list

def get_10Mb_matrix(input_matrix, chrom, segregation_chrom):
	chrom_sizes = {"chr1" : 195471971,"chr2" : 182113224,"chr3" : 160039680,"chr4" : 156508116, \
		"chr5" : 151834684,"chr6" : 149736546,"chr7" : 145441459,"chr8" : 129401213, \
		"chr9" : 124595110,"chr10" : 130694993, "chr11" : 122082543,"chr12" : 120129022, \
		"chr13" : 120421639,"chr14" : 124902244,"chr15" : 104043685,"chr16" : 98207768, \
		"chr17" : 94987271,"chr18" : 90702639,"chr19" : 61431566,"chrX": 171031299,"chrY" : 91744698}  
	value_list = []
	end_region = int(chrom_sizes[chrom])
	col1_name = segregation_chrom.columns[0]
	col2_name = segregation_chrom.columns[1]
	coordinate1 = col1_name.split(':')[1]
	coordinate2 = col2_name.split(':')[1]
	start1 = int(coordinate1.split('-')[0])
	start2 = int(coordinate2.split('-')[0])
	interval_size = int(start2 - start1)
	cut_off = int(10000000/int(interval_size))
	matrix_10Mb = np.ma.masked_invalid(input_matrix)
	for i in range(cut_off+1, len(matrix_10Mb)-1):
		diag = np.diagonal(matrix_10Mb, offset=i)
		fillNA_diag = np.nan 
		np.fill_diagonal(matrix_10Mb[:,i:], fillNA_diag)
	final_cutoff_matrix = matrix_10Mb + matrix_10Mb.T
	return final_cutoff_matrix


def get_top_bottom_top10per_DE_windows(DE_window_df):
	DE_window_matrix = DE_window_df.values
	DE_window_matrix_flat = np.matrix.flatten(DE_window_matrix)
	data_matrix = DE_window_matrix_flat[~np.isnan(DE_window_matrix_flat)]
	(mu_data, sigma_data) = norm.fit(data_matrix)
	more_df1 = norm.ppf(0.90, loc=mu_data, scale=sigma_data)
	more_df2 = norm.ppf(0.10, loc=mu_data, scale=sigma_data)	
	top_df1 = DE_window_df.copy()
	top_df2 = DE_window_df.copy()
	top_df1[top_df1 <= more_df1] = np.nan
	top_df2[top_df2 > more_df2] = np.nan
	top_df1_noNaN = top_df1.dropna()
	top_df2_noNaN = top_df2.dropna()
	top_df1_noNaN.columns = ['Differential_Top10%_Bins']
	top_df2_noNaN.columns = ['Differential_Top10%_Bins']
	return top_df1_noNaN, top_df2_noNaN
  
def Segregation_to_ZScore_differential(argv):
	"""
	Generate differential Z-Score matrices for a whole chromosome.
	:param ID_1,ID_2: dataset ID for naming the output file
	:param seg_table1: path to first input matrix or open python file
	:param seg_table2: path to second input matrix or open python file
	:param region: chromosome in format 'chr#'
	:param outfile: path to output directory
	"""	
	if len(argv)==6:
		(ID_1, ID_2, seg_table1, seg_table2, region, outfile) = argv
	else:
		print('Wrong number of arguments')
		sys.exit()
	
	#open segregation tables and subset chromosome
	open1 = open_segregation(seg_table1)
	open2 = open_segregation(seg_table2)
	segregation_chrom1 = open1.loc[region]
	segregation_chrom2 = open2.loc[region]
	segregation_matrix1 = segregation_chrom1.values
	segregation_matrix2 = segregation_chrom2.values
	
	#create bin names [chr:start-end]
	cnames = segregation_chrom1.index.to_frame(index=None)
	cnames["bin"]=region
	bins=cnames[("bin")].str.cat(cnames[("start")].astype(str), sep=":").str.cat(cnames[("stop")].astype(str), sep="-")

	#generate NPMI matrix for chromosome
	NPMI_matrix1 = calculate_NPMI(segregation_matrix1)
	NPMI_matrix2 = calculate_NPMI(segregation_matrix2)
	
	#generate list of windows to be masked in Z-Score matrix
	WDF_df1 = windows_detection_frequencies(open1)
	WDF_df2 = windows_detection_frequencies(open2)
	mask_indexlist1, mask_indexlist2 = get_bad_index_list_from_WDF(WDF_df1, WDF_df2)
	mask_indexlist1_chrom = [k for k in mask_indexlist1 if region+':' in k]
	mask_indexlist2_chrom = [k for k in mask_indexlist2 if region+':' in k]
	
	#set indices to be masked in Z-score matrices to NaN
	NPMI_dataframe1_formasking = pd.DataFrame(NPMI_matrix1)
	NPMI_dataframe2_formasking = pd.DataFrame(NPMI_matrix2)
	NPMI_dataframe1_formasking.columns=bins.astype(str)
	NPMI_dataframe2_formasking.columns=bins.astype(str)
	NPMI_dataframe1_formasking_final = NPMI_dataframe1_formasking.rename(index=bins.astype(str))
	NPMI_dataframe2_formasking_final = NPMI_dataframe2_formasking.rename(index=bins.astype(str))
	NPMI_dataframe1_formasking_final.loc[mask_indexlist1_chrom, :] = np.nan
	NPMI_dataframe2_formasking_final.loc[mask_indexlist2_chrom, :] = np.nan
	NPMI_masked1 = NPMI_dataframe1_formasking_final.values
	NPMI_masked2 = NPMI_dataframe2_formasking_final.values
		
	#generate ZScores from NPMI and take differential
	Zmatrix1 = zscore_matrix(NPMI_masked1)
	Zmatrix2 = zscore_matrix(NPMI_masked2)
	z_diff = Zmatrix1 - Zmatrix2

	#generate top 5% differential Z-Scores for each dataset
	top5_z_diff_Zmatrix1, top5_z_diff_Zmatrix2 = get_top5_bottom5_matrices(z_diff)

	#generate top 10% common contacts in differential Z-Scores
	common10_z_diff = get_common_top10_matrix (z_diff, Zmatrix1, Zmatrix2)

	#create dataframes for all of generated matrices
	NPMI_dataframe1 = pd.DataFrame(NPMI_matrix1)
	NPMI_dataframe2 = pd.DataFrame(NPMI_matrix2)
	Zmatrix1_dataframe = pd.DataFrame(Zmatrix1)
	Zmatrix2_dataframe = pd.DataFrame(Zmatrix2)
	z_diff_dataframe = pd.DataFrame(z_diff)
	top5_z_diff_Zmatrix1_dataframe = pd.DataFrame(top5_z_diff_Zmatrix1)
	top5_z_diff_Zmatrix2_dataframe = pd.DataFrame(top5_z_diff_Zmatrix2)
	common10_z_diff_dataframe = pd.DataFrame(common10_z_diff)

	#rename columns/index NPMI, ZScore, differential, top/common matrix files	
	NPMI_dataframe1.columns=bins.astype(str)
	NPMI_dataframe2.columns=bins.astype(str)
	Zmatrix1_dataframe.columns=bins.astype(str)
	Zmatrix2_dataframe.columns=bins.astype(str)	
	z_diff_dataframe.columns=bins.astype(str)
	top5_z_diff_Zmatrix1_dataframe.columns=bins.astype(str)
	top5_z_diff_Zmatrix2_dataframe.columns=bins.astype(str)
	NPMI_dataframe1_final = NPMI_dataframe1.rename(index=bins.astype(str))
	NPMI_dataframe2_final = NPMI_dataframe2.rename(index=bins.astype(str))
	Zmatrix1_final = Zmatrix1_dataframe.rename(index=bins.astype(str))
	Zmatrix2_final = Zmatrix2_dataframe.rename(index=bins.astype(str))
	z_diff_final = z_diff_dataframe.rename(index=bins.astype(str))
	top5_z_diff_Zmatrix1_dataframe_final = top5_z_diff_Zmatrix1_dataframe.rename(index=bins.astype(str))
	top5_z_diff_Zmatrix2_dataframe_final = top5_z_diff_Zmatrix2_dataframe.rename(index=bins.astype(str))
	common10_z_diff_dataframe_final = common10_z_diff_dataframe.rename(index=bins.astype(str))
	
	#generate differential windows from top5 differential list and get list of top 10% DE windows in each dataset
	diff_window_list = get_differential_Z_score_windowscore(top5_z_diff_Zmatrix1_dataframe, top5_z_diff_Zmatrix2_dataframe, region, NPMI_dataframe1_final)
	diff_index_list = list(NPMI_dataframe1_final.index)
	diff_window_dataframe_final = pd.DataFrame(diff_window_list, index=diff_index_list, columns=['Differential_Top_Bins'])
	top10per_DE_windows_df1, top10per_DE_windows_df2 = get_top_bottom_top10per_DE_windows(diff_window_dataframe_final)


	#Mask regions above 10Mb distance
	z_diff_final_short = get_10Mb_matrix(z_diff, region, NPMI_dataframe1_final)
	top5_z_diff_Zmatrix1_final_short = get_10Mb_matrix(top5_z_diff_Zmatrix1, region, NPMI_dataframe1_final)
	top5_z_diff_Zmatrix2_final_short = get_10Mb_matrix(top5_z_diff_Zmatrix2, region, NPMI_dataframe1_final)
	common10_z_diff_final_short = get_10Mb_matrix(common10_z_diff, region, NPMI_dataframe1_final)
	NPMImatrix1_short = get_10Mb_matrix(NPMI_matrix1, region, NPMI_dataframe1_final)
	NPMImatrix2_short = get_10Mb_matrix(NPMI_matrix2, region, NPMI_dataframe1_final)
	Zmatrix1_short = get_10Mb_matrix(Zmatrix1, region, NPMI_dataframe1_final)
	Zmatrix2_short = get_10Mb_matrix(Zmatrix2, region, NPMI_dataframe1_final)
	z_diff_short_dataframe = pd.DataFrame(z_diff_final_short)
	top5_z_diff_Zmatrix1_short_dataframe = pd.DataFrame(top5_z_diff_Zmatrix1_final_short)
	top5_z_diff_Zmatrix2_short_dataframe = pd.DataFrame(top5_z_diff_Zmatrix2_final_short)
	common10_z_diff_short_dataframe = pd.DataFrame(common10_z_diff_final_short)
	NPMImatrix1_short_dataframe = pd.DataFrame(NPMImatrix1_short)
	NPMImatrix2_short_dataframe = pd.DataFrame(NPMImatrix2_short)
	Zmatrix1_short_dataframe = pd.DataFrame(Zmatrix1_short)
	Zmatrix2_short_dataframe = pd.DataFrame(Zmatrix2_short)
	z_diff_short_dataframe.columns=bins.astype(str)
	top5_z_diff_Zmatrix1_short_dataframe.columns=bins.astype(str)
	top5_z_diff_Zmatrix2_short_dataframe.columns=bins.astype(str)
	common10_z_diff_short_dataframe.columns=bins.astype(str)
	NPMImatrix1_short_dataframe.columns=bins.astype(str)
	NPMImatrix2_short_dataframe.columns=bins.astype(str)
	Zmatrix1_short_dataframe.columns=bins.astype(str)
	Zmatrix2_short_dataframe.columns=bins.astype(str)
	z_diff_short_dataframe_final = z_diff_short_dataframe.rename(index=bins.astype(str))
	top5_z_diff_Zmatrix1_short_dataframe_final = top5_z_diff_Zmatrix1_short_dataframe.rename(index=bins.astype(str))
	top5_z_diff_Zmatrix2_short_dataframe_final = top5_z_diff_Zmatrix2_short_dataframe.rename(index=bins.astype(str))
	common10_z_diff_short_dataframe_final = common10_z_diff_short_dataframe.rename(index=bins.astype(str))
	NPMImatrix1_short_dataframe_final = NPMImatrix1_short_dataframe.rename(index=bins.astype(str))
	NPMImatrix2_short_dataframe_final = NPMImatrix2_short_dataframe.rename(index=bins.astype(str))
	Zmatrix1_short_dataframe_final = Zmatrix1_short_dataframe.rename(index=bins.astype(str))
	Zmatrix2_short_dataframe_final = Zmatrix2_short_dataframe.rename(index=bins.astype(str))


	#generate differential windows with 10Mb cut-off from top5 differential list and get list of top 10% DE windows in each dataset
	diff_window_list_10Mb = get_differential_Z_score_windowscore(top5_z_diff_Zmatrix1_short_dataframe, top5_z_diff_Zmatrix2_short_dataframe, region, NPMImatrix1_short_dataframe)
	diff_index_list_10Mb = list(NPMImatrix1_short_dataframe.index)
	diff_window_10Mb_dataframe_final = pd.DataFrame(diff_window_list_10Mb, index=diff_index_list, columns=['Differential_Top_Bins'])
	top10per_DE_windows_short_df1, top10per_DE_windows_short_df2 = get_top_bottom_top10per_DE_windows(diff_window_10Mb_dataframe_final)

		
	#save all files
	#NPMI_dataframe1_final.to_csv(outfile + str(ID_1) + '_' + region + '_NPMI.txt.gz',
	#	sep="\t",
	#	index=True,
	#	header=True,
	#	compression='gzip')	
	#NPMI_dataframe2_final.to_csv(outfile + str(ID_2) + '_' + region + '_NPMI.txt.gz',
	#	sep="\t",
	#	index=True,
	#	header=True,
	#	compression='gzip')	
	#Zmatrix1_final.to_csv(outfile + str(ID_1) + '_' + region + '_ZScore.txt.gz',
	#	sep="\t",
	#	index=True,
	#	header=True,
	#	compression='gzip')	
	#Zmatrix2_final.to_csv(outfile + str(ID_2) + '_' + region + '_ZScore.txt.gz',
	#	sep="\t",
	#	index=True,
	#	header=True,
	#	compression='gzip')
	#z_diff_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_ZScore_diff.txt.gz',
	#	sep="\t",
	#	index=True,
	#	header=True,
	#	compression='gzip')
	#top5_z_diff_Zmatrix1_dataframe_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_ZScore_diff_top5%_' + str(ID_1) + '.txt.gz',
	#	sep="\t",
	#	index=True,
	#	header=True,
	#	compression='gzip')
	#top5_z_diff_Zmatrix2_dataframe_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_ZScore_diff_top5%_' + str(ID_2) + '.txt.gz',
	#	sep="\t",
	#	index=True,
	#	header=True,
	#	compression='gzip')
	#common10_z_diff_dataframe_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_ZScore_diff_common_top10%.txt.gz',
	#	sep="\t",
	#	index=True,
	#	header=True,
	#	compression='gzip')
	diff_window_dataframe_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_differentialwindows.txt',
		sep="\t",
		index=True,
		header=True)
	top10per_DE_windows_df1.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_differentialwindows_top10per_' + str(ID_1) + '.txt',
		sep="\t",
		index=True,
		header=True)
	top10per_DE_windows_df2.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_differentialwindows_top10per_' + str(ID_2) + '.txt',
		sep="\t",
		index=True,
		header=True)
	diff_window_10Mb_dataframe_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_differentialwindows_10Mbcutoff.txt',
		sep="\t",
		index=True,
		header=True)
	top10per_DE_windows_short_df1.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_differentialwindows_top10per_' + str(ID_1) + '_10Mbcutoff.txt',
		sep="\t",
		index=True,
		header=True)
	top10per_DE_windows_short_df2.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_differentialwindows_top10per_' + str(ID_2) + '_10Mbcutoff.txt',
		sep="\t",
		index=True,
		header=True)
	z_diff_short_dataframe_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_ZScore_diff_10Mbcutoff.txt.gz',
		sep="\t",
		index=True,
		header=True,
		compression='gzip')
	top5_z_diff_Zmatrix1_short_dataframe_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_ZScore_diff_top5%_10Mbcutoff_' + str(ID_1) + '.txt.gz',
		sep="\t",
		index=True,
		header=True,
		compression='gzip')
	top5_z_diff_Zmatrix2_short_dataframe_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_ZScore_diff_top5%_10Mbcutoff_' + str(ID_2) + '.txt.gz',
		sep="\t",
		index=True,
		header=True,
		compression='gzip')
	NPMImatrix1_short_dataframe_final.to_csv(outfile + str(ID_1) + '_' + region + '_NPMI_10Mbcutoff.txt.gz',
		sep="\t",
		index=True,
		header=True,
		compression='gzip')
	NPMImatrix2_short_dataframe_final.to_csv(outfile + str(ID_2) + '_' + region + '_NPMI_10Mbcutoff.txt.gz',
		sep="\t",
		index=True,
		header=True,
		compression='gzip')
	Zmatrix1_short_dataframe_final.to_csv(outfile + str(ID_1) + '_' + region + '_ZScore_10Mbcutoff.txt.gz',
		sep="\t",
		index=True,
		header=True,
		compression='gzip')
	Zmatrix2_short_dataframe_final.to_csv(outfile + str(ID_2) + '_' + region + '_ZScore_10Mbcutoff.txt.gz',
		sep="\t",
		index=True,
		header=True,
		compression='gzip')
	common10_z_diff_short_dataframe_final.to_csv(outfile + str(ID_1) + '_' + str(ID_2) + '_' + region + '_ZScore_diff_common_top10%_10Mbcutoff.txt.gz',
		sep="\t",
		index=True,
		header=True,
		compression='gzip')
	print(region + ' done!')
	pass

#%%

Segregation_to_ZScore_differential(sys.argv[1:])
