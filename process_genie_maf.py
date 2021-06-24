import pandas as pd
import numpy as np
import pickle
import pyranges as pr
from ATGC.model.CustomKerasModels import InputFeatures, ATGC
from ATGC.model.CustomKerasTools import BatchGenerator, Losses, histogram_equalization
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[6], True)
tf.config.experimental.set_visible_devices(physical_devices[6], 'GPU')


##your path to the files directory
path = 'ATGC/files/'

usecols = ['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'Variant_Classification', 'Variant_Type', 'Reference_Allele', 'Tumor_Seq_Allele2',  'Tumor_Sample_Barcode', 't_ref_count', 't_alt_count']


##your GENIE MAF
genie_maf = pd.read_csv('data_mutations_extended_8.3-consortium.txt', sep='\t',
                        usecols=usecols,
                        low_memory=False)

##your GENIE samples
genie_sample_table = pd.read_csv('tumor_normal.csv', sep=',', low_memory=False)
genie_sample_table.rename(columns={'sample_id': 'Tumor_Sample_Barcode'}, inplace=True)


genie_maf = genie_maf.loc[genie_maf['Tumor_Sample_Barcode'].isin(genie_sample_table['Tumor_Sample_Barcode'])]

genie_maf.reset_index(inplace=True, drop=True)


path_to_genome = path + 'chromosomes/'
chromosomes = {}
for i in list(range(1, 23))+['X', 'Y']:
    with open(path_to_genome+'/'+'chr'+str(i)+'.txt') as f:
        chromosomes[str(i)] = f.read()


##Use GFF3 to annotate variants
##ftp://ftp.ensembl.org/pub/grch37/current/gff3/homo_sapiens/
gff = pd.read_csv(path + 'Homo_sapiens.GRCh37.87.gff3',
                  sep='\t',
                  names=['chr', 'unknown', 'gene_part', 'start', 'end', 'unknown2', 'strand', 'unknown3', 'gene_info'],
                  usecols=['chr','gene_part', 'start', 'end', 'gene_info'],
                  low_memory=False)


gff_cds_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'CDS') & gff['chr'].isin(chromosomes), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge()
gff_exon_pr = pr.PyRanges(gff.loc[(gff['gene_part'] == 'exon') & gff['chr'].isin(chromosomes), ['chr', 'start', 'end', 'gene_info']].astype({'start': int, 'end': int}).rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'})).merge()
del gff

##make index column for merging
genie_maf['index'] = genie_maf.index.values

maf_pr = pr.PyRanges(genie_maf.loc[:, ['Chromosome', 'Start_Position', 'End_Position', 'index']].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'}))

##used genie 8.3 panels, panel information can be obtained from https://www.synapse.org/#!Synapse:syn7844529
genie = pd.read_csv('genomic_information_8.3-consortium.txt', sep='\t', low_memory=False)
panels = genie.SEQ_ASSAY_ID.unique()
panel_df = pd.DataFrame(data=panels, columns=['Panel'])


total_sizes = []
cds_sizes = []
exon_sizes = []
panel_prs = []

for panel in panels:
    if panel in genie_sample_table['seq_assay_id'].unique():
        print(panel)
        panel_pr = pr.PyRanges(genie.loc[(genie['SEQ_ASSAY_ID'] == panel) & genie['Chromosome'].isin(chromosomes), 'Chromosome':'End_Position'].rename(columns={'Start_Position': 'Start', 'End_Position': 'End'})).merge()
        total_sizes.append(sum([i + 1 for i in panel_pr.lengths()]))
        cds_sizes.append(sum([i + 1 for i in panel_pr.intersect(gff_cds_pr).lengths()]))
        exon_sizes.append(sum([i + 1 for i in panel_pr.intersect(gff_exon_pr).lengths()]))
        panel_prs.append(panel_pr)


grs = {k: v for k, v in zip(['CDS', 'exon'] + list(panels[np.isin(panels, genie_sample_table['seq_assay_id'].unique())]), [gff_cds_pr, gff_exon_pr] + panel_prs)}
result = pr.count_overlaps(grs, pr.concat({'maf': maf_pr}.values()))
result = result.df

genie_maf = pd.merge(genie_maf, result.iloc[:, 3:], how='left', on='index')


genie_maf = pd.merge(genie_maf, genie_sample_table, on='Tumor_Sample_Barcode')


def variant_features(maf, ref_length=6, alt_length=6, five_p_length=11, three_p_length=11):
    refs = []
    alts = []
    five_ps = []
    three_ps = []
    if ref_length % 2 != 0:
        ref_length += 1
        print('Your ref length was not even, incrementing by 1.')
    if alt_length % 2 != 0:
        alt_length += 1
        print('Your alt length was not even, incrementing by 1.')

    for index, row in enumerate(maf.itertuples()):
        Ref = row.Reference_Allele
        Alt = row.Tumor_Seq_Allele2
        Chr = str(row.Chromosome)
        Start = row.Start_Position
        End = row.End_Position
        if pd.isna(Alt):
            print(str(index)+' Alt is nan')
            Ref = np.nan
            Alt = np.nan
            context_5p = np.nan
            context_3p = np.nan
        else:
            if len(Ref) > ref_length:
                Ref = Ref[:int(ref_length / 2)] + Ref[-int(ref_length / 2):]
            else:
                while len(Ref) < ref_length:
                    Ref += '-'
            if len(Alt) > alt_length:
                Alt = Alt[:int(alt_length / 2)] + Alt[-int(alt_length / 2):]
            else:
                while len(Alt) < alt_length:
                    Alt += '-'
            if row.Reference_Allele == '-':
                ##the TCGA coordinates for a null ref are a little weird
                assert Start-five_p_length >= 0
                context_5p = chromosomes[Chr][Start-five_p_length:Start]
                context_3p = chromosomes[Chr][Start:Start+three_p_length]
            else:
                assert Start-(five_p_length+1) >= 0
                context_5p = chromosomes[Chr][Start-(five_p_length+1):Start-1]
                context_3p = chromosomes[Chr][End:End+three_p_length]
        refs.append(Ref)
        alts.append(Alt)
        five_ps.append(context_5p)
        three_ps.append(context_3p)
    return refs, alts, five_ps, three_ps

genie_maf['Ref'], genie_maf['Alt'], genie_maf['five_p'], genie_maf['three_p'] = variant_features(genie_maf)

genie_maf.drop(columns=['index'], inplace=True)
non_syn_models = pickle.load(open('figures/tmb/genie/results/non_syn.pkl', 'rb'))
cds_models =  pickle.load(open('figures/tmb/genie/results/cds.pkl', 'rb'))
non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']




results = {}
for panel in genie_sample_table['seq_assay_id'].unique():
    print(panel)
    panel_maf = genie_maf.loc[genie_maf['seq_assay_id'] == panel]
    if panel =='UCSF-NIMV4-TN':
        weights = cds_models[panel]['weights']
        panel_maf = panel_maf.loc[(panel_maf['CDS'] > 0) & (panel_maf[panel] > 0)]
    else:
        weights = non_syn_models[panel]['weights']
        panel_maf = panel_maf.loc[(genie_maf['Variant_Classification'].isin(non_syn)) & (panel_maf[panel] > 0)]

    panel_maf.reset_index(inplace=True, drop=True)
    panel_samples = genie_sample_table.loc[genie_sample_table['seq_assay_id'] == panel]
    panel_samples.reset_index(inplace=True, drop=True)
    ##create a new column called index that is the sample idxs
    panel_maf = pd.merge(panel_maf, panel_samples.Tumor_Sample_Barcode.reset_index(), how='left', on='Tumor_Sample_Barcode')

    ##if you want to check indexes match up
    maf_indexes = {i: j for i, j in zip(panel_maf['index'].values, panel_maf['Tumor_Sample_Barcode'].values)}
    sample_indexes = {i: j for i, j in zip(panel_samples.index.values, panel_samples['Tumor_Sample_Barcode'].values)}
    X = True
    for index in maf_indexes:
        if maf_indexes[index] != sample_indexes[index]:
            X = False
    print(X)

    samples_idx = panel_maf['index'].values

    # 5p, 3p, ref, alt
    nucleotide_mapping = {'-': 0, 'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
    seqs_5p = np.stack(panel_maf.five_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x[-6:]])).values, axis=0)
    seqs_3p = np.stack(panel_maf.three_p.apply(lambda x: np.array([nucleotide_mapping[i] for i in x[:6]])).values, axis=0)
    seqs_ref = np.stack(panel_maf.Ref.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)
    seqs_alt = np.stack(panel_maf.Alt.apply(lambda x: np.array([nucleotide_mapping[i] for i in x])).values, axis=0)

    ##map genomic coordinates to the exome
    # chr, pos
    chromosome_mapping = dict(zip([str(i) for i in list(range(1, 23))] + ['X', 'Y'], list(range(1, 25))))
    gen_chr = np.array([chromosome_mapping[i] for i in panel_maf.Chromosome.values])
    with open(path + 'chr_sizes.tsv') as f:
        sizes = [i.split('\t') for i in f.read().split('\n')[:-1]]

    chromosome_sizes = {i: float(j) for i, j in sizes}
    gen_pos = panel_maf['Start_Position'].values / [chromosome_sizes[i] for i in panel_maf.Chromosome.values]
    cds = np.ones(len(gen_pos))
    D = {'sample_idx': samples_idx,
                 'seq_5p': seqs_5p,
                 'seq_3p': seqs_3p,
                 'seq_ref': seqs_ref,
                 'seq_alt': seqs_alt,
                 'chr': gen_chr,
                 'pos_float': gen_pos,
                 'cds': cds}

    variant_encoding = np.array([0, 2, 1, 4, 3])
    D['seq_5p'] = np.stack([D['seq_5p'], variant_encoding[D['seq_3p'][:, ::-1]]], axis=2)
    D['seq_3p'] = np.stack([D['seq_3p'], variant_encoding[D['seq_5p'][:, :, 0][:, ::-1]]], axis=2)
    t = D['seq_ref'].copy()
    i = t != 0
    t[i] = variant_encoding[D['seq_ref'][:, ::-1]][i[:, ::-1]]
    D['seq_ref'] = np.stack([D['seq_ref'], t], axis=2)
    t = D['seq_alt'].copy()
    i = t != 0
    t[i] = variant_encoding[D['seq_alt'][:, ::-1]][i[:, ::-1]]
    D['seq_alt'] = np.stack([D['seq_alt'], t], axis=2)
    del i, t

    D['strand'] = np.ones(len(gen_pos))

    D['pos_float'] = np.ones(len(D['pos_float']))

    features = [InputFeatures.OnesLike({'position': D['pos_float'][:, np.newaxis]})]
    sample_features = ()

    y_label = np.ones(len(panel_samples))[:, np.newaxis]

    atgc = ATGC(features, aggregation_dimension=64, fusion_dimension=32, sample_features=sample_features)
    atgc.build_instance_encoder_model(return_latent=False)
    atgc.build_sample_encoder_model()
    atgc.build_mil_model(output_dim=8, output_extra=1, output_type='quantiles', aggregation='recursion', mil_hidden=(16,))
    atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=Losses.Weighted.QuantileLoss.quantile_loss)
    atgc.mil_model.set_weights(weights)

    data = BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                          y_label=y_label, sampling_approach=None).data_generator()
    test_data = next(data)


    predictions = atgc.mil_model.predict(test_data[0])[0, :, :-1]

    if panel == 'UCSF-NIMV4-TN':
        panel_counts = panel_maf[['CDS', 'Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x), (x['CDS'] > 0).sum()], index=['panel_all_counts', 'panel_cds_counts']))
    else:
        panel_counts = panel_maf[['Variant_Classification', 'Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x), (x['Variant_Classification'].isin(non_syn)).sum()], index=['panel_all_counts', 'panel_non_syn_counts']))

    panel_samples = pd.merge(panel_samples, panel_counts, how='left', on='Tumor_Sample_Barcode')
    if panel == 'UCSF-NIMV4-TN':
        panel_samples.fillna({'panel_cds_counts': 0}, inplace=True)
    else:
        panel_samples.fillna({'panel_non_syn_counts': 0}, inplace=True)

    if panel == 'UCSF-NIMV4-TN':
        results[panel] = {'predictions': predictions, 'panel_counts': panel_samples.panel_cds_counts.values, 'samples': panel_samples.Tumor_Sample_Barcode.values}
    else:
        results[panel] = {'predictions': predictions, 'panel_counts': panel_samples.panel_non_syn_counts.values, 'samples': panel_samples.Tumor_Sample_Barcode.values}



##if you want to look at the predictions
# import pylab as plt
#
# plt.scatter(results['UCSF-NIMV4-TN']['panel_counts'], results['UCSF-NIMV4-TN']['predictions'][:, 1] , color='k', s=.1)
# plt.scatter(results['UCSF-NIMV4-TN']['panel_counts'], results['UCSF-NIMV4-TN']['predictions'][:, 0], color='k', s=.1)
# plt.scatter(results['UCSF-NIMV4-TN']['panel_counts'], results['UCSF-NIMV4-TN']['predictions'][:, 2], color='k', s=.1)
# plt.yticks([np.log(i+1) for i in [0, 1, 2, 3, 5, 10, 25, 64]], ['0', '1', '2', '3', '5', '10', '25', '64'], fontsize=9)
# plt.title('UCSF-NIMV4-TN CDS')
# plt.xlabel('Panel Counts')
# plt.ylabel('TMB Prediction')
#
# plt.clf()
# plt.scatter(results['MSK-IMPACT468']['panel_counts'], results['MSK-IMPACT468']['predictions'][:, 1] , color='k', s=.1)
# plt.scatter(results['MSK-IMPACT468']['panel_counts'], results['MSK-IMPACT468']['predictions'][:, 0], color='k', s=.1)
# plt.scatter(results['MSK-IMPACT468']['panel_counts'], results['MSK-IMPACT468']['predictions'][:, 2], color='k', s=.1)
# plt.yticks([np.log(i+1) for i in [0, 1, 2, 3, 5, 10, 25, 64]], ['0', '1', '2', '3', '5', '10', '25', '64'], fontsize=9)
# plt.title('MSK-IMPACT468 Nonsyn')
# plt.xlabel('Panel Counts')
# plt.ylabel('TMB Prediction')
#
#
# plt.clf()

# plt.scatter(results['MSK-IMPACT410']['panel_counts'], results['MSK-IMPACT410']['predictions'][:, 1] , color='k', s=.1)
# plt.scatter(results['MSK-IMPACT410']['panel_counts'], results['MSK-IMPACT410']['predictions'][:, 0], color='k', s=.1)
# plt.scatter(results['MSK-IMPACT410']['panel_counts'], results['MSK-IMPACT410']['predictions'][:, 2], color='k', s=.1)
# plt.yticks([np.log(i+1) for i in [0, 1, 2, 3, 5, 10, 25, 64]], ['0', '1', '2', '3', '5', '10', '25', '64'], fontsize=9)
# plt.title('MSK-IMPACT410 Nonsyn')
# plt.xlabel('Panel Counts')
# plt.ylabel('TMB Prediction')
#
# plt.clf()

# plt.scatter(results['MSK-IMPACT341']['panel_counts'], results['MSK-IMPACT341']['predictions'][:, 1] , color='k', s=.1)
# plt.scatter(results['MSK-IMPACT341']['panel_counts'], results['MSK-IMPACT341']['predictions'][:, 0], color='k', s=.1)
# plt.scatter(results['MSK-IMPACT341']['panel_counts'], results['MSK-IMPACT341']['predictions'][:, 2], color='k', s=.1)
# plt.yticks([np.log(i+1) for i in [0, 1, 2, 3, 5, 10, 25, 64]], ['0', '1', '2', '3', '5', '10', '25', '64'], fontsize=9)
# plt.title('MSK-IMPACT341 Nonsyn')
# plt.xlabel('Panel Counts')
# plt.ylabel('TMB Prediction')

# plt.clf()

# plt.scatter(results['MSK-IMPACT-HEME-400']['panel_counts'], results['MSK-IMPACT-HEME-400']['predictions'][:, 1] , color='k', s=.1)
# plt.scatter(results['MSK-IMPACT-HEME-400']['panel_counts'], results['MSK-IMPACT-HEME-400']['predictions'][:, 0], color='k', s=.1)
# plt.scatter(results['MSK-IMPACT-HEME-400']['panel_counts'], results['MSK-IMPACT-HEME-400']['predictions'][:, 2], color='k', s=.1)
# plt.yticks([np.log(i+1) for i in [0, 1, 2, 3, 5, 10, 25, 64]], ['0', '1', '2', '3', '5', '10', '25', '64'], fontsize=9)
# plt.title('MSK-IMPACT-HEME-400 Nonsyn')
# plt.xlabel('Panel Counts')
# plt.ylabel('TMB Prediction')


all_samples = np.concatenate([results[i]['samples'] for i in results])
all_counts = np.concatenate([results[i]['panel_counts'] for i in results])
lower_bounds = np.exp(np.concatenate([results[i]['predictions'][:, 0] for i in results])) - 1
medians = np.exp(np.concatenate([results[i]['predictions'][:, 1] for i in results])) - 1
upper_bounds = np.exp(np.concatenate([results[i]['predictions'][:, 2] for i in results])) - 1


mask = medians > 64
medians = medians.astype(str)
medians[mask] = '>64'
lower_bounds = lower_bounds.astype(str)
lower_bounds[mask] = 'nan'
upper_bounds = upper_bounds.astype(str)
upper_bounds[mask] = 'nan'

predictions_dataframe = pd.DataFrame(data=np.array([all_samples, all_counts, lower_bounds, medians, upper_bounds]).T,
                                     columns=['Tumor_Sample_Barcode', 'Panel_Counts', 'Lower_Estimate', 'Estimate', 'Upper_Estimate'])


genie_sample_table = pd.merge(genie_sample_table, predictions_dataframe, on='Tumor_Sample_Barcode')
genie_sample_table.rename(columns={'Tumor_Sample_Barcode': 'sample_id'}, inplace=True)

final_table = genie_sample_table.loc[~(genie_sample_table['seq_assay_id'] == 'UHN-48-V1')]  ##small panel

final_table.to_csv('tumor_normal_predictions.csv', index=False)