import re
import json
import numpy as np
import pickle
import pandas as pd
from ATGC.model.CustomKerasModels import InputFeatures, ATGC
from ATGC.model.CustomKerasTools import BatchGenerator, Losses, histogram_equalization
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from sklearn.metrics import r2_score
disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[6], True)
tf.config.experimental.set_visible_devices(physical_devices[6], 'GPU')

##path to files
path = 'files/'
genie_path = 'your path to data'
tcga_maf = pickle.load(open(genie_path + 'tcga_maf_table.pkl', 'rb'))
samples = pickle.load(open(genie_path + 'tcga_sample_table.pkl', 'rb'))
panels = pickle.load(open(genie_path + 'tcga_panel_table.pkl', 'rb'))

pcawg_maf = pickle.load(open(genie_path + 'pcawg_maf_table.pkl', 'rb'))
pcawg_samples = pickle.load(open(genie_path + 'pcawg_sample_table.pkl', 'rb'))



non_syn = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Nonstop_Mutation']
tcga_maf = tcga_maf.loc[tcga_maf['Variant_Classification'].isin(non_syn)]
pcawg_maf = pcawg_maf.loc[pcawg_maf['Variant_Classification'].isin(non_syn)]
pcawg_maf.rename(columns={'Start_position': 'Start_Position', 'End_position': 'End_Position'}, inplace=True)


maf = tcga_maf.append(pcawg_maf)


##fill in some missing cancer labels
with open(path + 'cases.2020-02-28.json', 'r') as f:
    tcga_cancer_info = json.load(f)
cancer_labels = {i['submitter_id']: i['project']['project_id'].split('-')[-1] for i in tcga_cancer_info}
cancer_labels['TCGA-AB-2852'] = 'LAML'
samples['type'] = samples['bcr_patient_barcode'].apply(lambda x: cancer_labels[x])

##remove samples without a kit that covered the exome
samples_covered = samples.loc[samples['Exome_Covered']]
samples_unknown = samples.loc[(samples['Exome_Unknown']) & (samples['type'].isin(['KIRC', 'BRCA']))]
samples = samples_covered.append(samples_unknown)

samples = samples.append(pcawg_samples)


##remove samples with TMB above 64
samples = samples.loc[samples['non_syn_counts']/31.8 < 64]

samples.reset_index(inplace=True, drop=True)

##limit MAF to samples
maf = maf.loc[maf['Tumor_Sample_Barcode'].isin(samples.Tumor_Sample_Barcode.values)]

##your file of GENIE samples and their panel
test_samples = pd.read_csv('tumor_normal.csv', sep=',')
to_use = test_samples['seq_assay_id'].unique()

results = {}
for panel in panels['Panel'].values:
    if panel in to_use:
        print(panel)
        panel_maf = maf.loc[maf[panel] > 0]
        panel_maf.reset_index(inplace=True, drop=True)

        ##create a new column called index that is the sample idxs
        panel_maf = pd.merge(panel_maf, samples.Tumor_Sample_Barcode.reset_index(), how='left', on='Tumor_Sample_Barcode')

        ##if you want to check indexes match up
        maf_indexes = {i: j for i, j in zip(panel_maf['index'].values, panel_maf['Tumor_Sample_Barcode'].values)}
        sample_indexes = {i: j for i, j in zip(samples.index.values, samples['Tumor_Sample_Barcode'].values)}
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
        cds = panel_maf['CDS_position'].astype(str).apply(lambda x: (int(x) % 3) + 1 if re.match('^[0-9]+$', x) else 0).values

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

        D['strand'] = panel_maf['STRAND'].astype(str).apply(lambda x: {'nan': 0, '-1.0': 1, '1.0': 2}[x]).values

        D['pos_float'] = np.ones(len(D['pos_float']))

        features = [InputFeatures.OnesLike({'position': D['pos_float'][:, np.newaxis]})]
        sample_features = ()

        y_label = np.log(samples.non_syn_counts.values / (panels.loc[panels['Panel'] == 'Agilent_kit']['cds'].values[0] / 1e6) + 1)[:, np.newaxis]
        y_label = np.repeat(y_label, 3, axis=-1)

        runs = 3
        initial_weights = []
        metrics = [Losses.Weighted.QuantileLoss.quantile_loss]
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=40, mode='min', restore_best_weights=True)]

        for i in range(runs):
            atgc = ATGC(features, aggregation_dimension=64, fusion_dimension=32, sample_features=sample_features)
            atgc.build_instance_encoder_model(return_latent=False)
            atgc.build_sample_encoder_model()
            atgc.build_mil_model(output_dim=8, output_extra=1, output_type='quantiles', aggregation='recursion', mil_hidden=(16,))
            atgc.mil_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=Losses.Weighted.QuantileLoss.quantile_loss, metrics=metrics)
            initial_weights.append(atgc.mil_model.get_weights())

        data = BatchGenerator(x_instance_sample_idx=D['sample_idx'], x_instance_features=features, x_sample=sample_features,
                              y_label=y_label, sampling_approach=None).data_generator()
        eval_data = next(data)

        eval = 100
        for initial_weight in initial_weights:
            atgc.mil_model.set_weights(initial_weight)
            atgc.mil_model.fit(data,
                               steps_per_epoch=2,
                               epochs=10000,
                               shuffle=False,
                               callbacks=callbacks)
            run_eval = atgc.mil_model.evaluate(eval_data[0], eval_data[1])[1]
            if run_eval < eval:
                eval = run_eval
                best_weights = atgc.mil_model.get_weights()

        atgc.mil_model.set_weights(best_weights)
        predictions = atgc.mil_model.predict(eval_data[0])[0, :, :-1]
        mse = np.mean(((y_label[:, 0] - predictions[:, 1])**2))
        mae = np.mean(np.absolute(y_label[:, 0] - predictions[:, 1]))
        r2 = r2_score(y_label[:, 0], predictions[:, 1])

        panel_counts = panel_maf[['Variant_Classification', 'Tumor_Sample_Barcode']].groupby('Tumor_Sample_Barcode').apply(lambda x: pd.Series([len(x), (x['Variant_Classification'].isin(non_syn)).sum()], index=['panel_all_counts', 'panel_non_syn_counts']))
        panel_samples = pd.merge(samples, panel_counts, how='left', on='Tumor_Sample_Barcode')
        panel_samples.fillna({'panel_non_syn_counts': 0}, inplace=True)

        results[panel] = {'predictions': predictions, 'panel_counts': panel_samples.panel_non_syn_counts.values, 'y_true': y_label[:, 0], 'weights': best_weights, 'metrics': [mse, mae, r2]}


with open('non_syn.pkl', 'wb') as f:
    pickle.dump(results, f)