""""
scTDA. Library for topological data analysis of high-throughput single-cell RNA-seq data.

Copyright 2017, Pablo G. Camara. All rights reserved.

"""

__author__ = "Pablo G. Camara"
__maintainer__ = "Pablo G. Camara"
__email__ = "pablo.g.camara@gmail.com"
__credits__ = "Patrick van Nieuwenhuizen, Luis Aparicio, Yan Meng"


import json
import matplotlib_venn
import networkx
import numexpr
import numpy
import numpy.linalg
import numpy.random
import pandas
import pickle
import pylab
import requests
import sakmapper
import scipy.cluster.hierarchy as sch
import scipy.interpolate
import scipy.optimize
import scipy.signal
import scipy.spatial.distance
import scipy.stats
import sklearn.cluster
import sklearn.metrics.pairwise
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")
pylab.rcParams["patch.force_edgecolor"] = True
pylab.rcParams['patch.facecolor'] = 'k'


"""
GLOBAL METHODS
"""


def ParseAyasdiGraph(name, source, lab, user, password):
    """
    Parses Ayasdi graph given by 'source' ID and 'lab' ID, and generates files name.gexf, name.json and
    name.groups.json that are used by scTDA classes. Arguments 'user' and 'password' specify Ayasdi login credentials.
    """
    session = requests.Session()
    session.post('https://platform.ayasdi.com/workbench/login', data={'username': user, 'passphrase': password},
                 verify=False)
    output_network = session.get('https://platform.ayasdi.com/workbench/v0/sources/' + source + '/networks/' + lab)
    network = json.loads(output_network.content)
    nodes_ids = [int(node['id']) for node in network['nodes']]
    nodes_contents = {}
    for node_id in nodes_ids:
        output_node = session.post('https://platform.ayasdi.com/workbench/v0/sources/' + source +
                                   '/retrieve_row_indices',
                         data=json.dumps({"network_nodes_descriptions": [{"network_id": lab, "node_ids": [node_id]}]}),
                                      headers={"Content-type": "application/json"})
        nodes_contents[node_id] = json.loads(output_node.content)['row_indices']
    with open(name + '.json', 'wb') as network_json_file:
        json.dump(nodes_contents, network_json_file)
    nodes_sizes = []
    with open(name + '.gexf', 'w') as network_gexf_file:
        network_gexf_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        network_gexf_file.write('<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n')
        network_gexf_file.write('\t<graph mode="static" defaultedgetype="undirected">\n')
        network_gexf_file.write('\t\t<nodes>\n')
        for nod in network['nodes']:
            network_gexf_file.write('\t\t\t<node id="' + str(nod['id']) +
                                    '" label="' + str(nod['row_count']) + '" />\n')
            nodes_sizes.append(float(nod['row_count']))
        network_gexf_file.write('\t\t</nodes>\n')
        network_gexf_file.write('\t\t<edges>\n')
        for edge_index, edge in enumerate(network['links']):
            network_gexf_file.write('\t\t\t<edge id="' + str(edge_index) + '" source="' + str(edge['from']) +
                                    '" target="' + str(edge['to']) + '" />\n')
        network_gexf_file.write('\t\t</edges>\n')
        network_gexf_file.write('\t</graph>\n')
        network_gexf_file.write('</gexf>\n')
    output_node_groups = session.get('https://platform.ayasdi.com/workbench/v0/sources/' + source +
                                     '/networks/' + lab + '/node_groups')
    node_groups = json.loads(output_node_groups.content)
    groups = {}
    for m in node_groups:
        groups[m['name']] = m['node_ids']
    with open(name + '.groups.json', 'wb') as network_groups_file:
        json.dump(groups, network_groups_file)


def benjamini_hochberg(pvalues):
    """
    Benjamini-Hochberg adjusted p-values for multiple testing. 'pvalues' contains a list of p-values. Adapted from
    http://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python. Not subjected
    to copyright.
    """
    p_values = numpy.array(pvalues)
    number_of_p_values = float(p_values.shape[0])
    q_values = numpy.empty(int(number_of_p_values))
    pairs = [(p_value, p_value_index) for p_value_index, p_value in enumerate(p_values)]
    pairs.sort()
    pairs.reverse()
    pre_q_values = []
    for pair_index, pair in enumerate(pairs):
        rank = number_of_p_values - pair_index
        p_value, _ = pair
        pre_q_values.append((number_of_p_values/rank) * p_value)
    for q_value_index in xrange(0, int(number_of_p_values)-1):
        if pre_q_values[q_value_index] < pre_q_values[q_value_index+1]:
            pre_q_values[q_value_index+1] = pre_q_values[q_value_index]
    for pair_index, pair in enumerate(pairs):
        p_value, index = pair
        q_values[index] = pre_q_values[pair_index]
    return list(q_values)


def is_number(s):
    """
    Checks whether a string can be converted into a float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def compare_results(table1, table2, threshold=0.05):
    """
    Compares two tables produced by UnrootedGraph.save() or RootedGraph.save(). The parameter 'threshold' indicates
    the threshold in the q-value for a gene to be considered as significant. It produces the Venn diagram comparing
    genes that are significant in each of the analysis. If the input tables were produced by RootedGraph.save()
    it also plots the correlation of the centroid of the genes that are significant in the two data sets, as well as
    the correlation of the dispersion of the genes that are significant in both data sets.
    """
    significant_genes_network1 = []
    significant_genes_centroids_network1 = {}
    significant_genes_dispersions_network1 = {}
    file_network1 = open(table1 + '.genes.tsv', 'r')
    genes = []
    is_rooted = False
    for line_index, line in enumerate(file_network1):
        line_split = line[:-1].split('\t')
        if line_index == 0:
            if 'Centroid' in line_split:
                is_rooted = True
            else:
                is_rooted = False
        else:
            genes.append(line_split[0])
            if float(line_split[7]) < threshold:
                significant_genes_network1.append(line_split[0])
                if is_rooted:
                    significant_genes_centroids_network1[line_split[0]] = float(line_split[8])
                    significant_genes_dispersions_network1[line_split[0]] = float(line_split[9])
    file_network1.close()
    significant_genes_network2 = []
    common_significant_genes_centroids_network1 = []
    common_significant_genes_centroids_network2 = []
    common_significant_genes_dispersions_network1 = []
    common_significant_genes_dispersions_network2 = []
    file_network2 = open(table2 + '.genes.tsv', 'r')
    for line_index, line in enumerate(file_network2):
        if line_index > 0:
            line_split = line[:-1].split('\t')
            genes.append(line_split[0])
            if float(line_split[7]) < threshold:
                significant_genes_network2.append(line_split[0])
                if line_split[0] in significant_genes_network1 and is_rooted:
                    common_significant_genes_centroids_network1.append(
                        significant_genes_centroids_network1[line_split[0]])
                    common_significant_genes_dispersions_network1.append(
                        significant_genes_dispersions_network1[line_split[0]])
                    common_significant_genes_centroids_network2.append(float(line_split[8]))
                    common_significant_genes_dispersions_network2.append(float(line_split[9]))
    file_network2.close()
    genes = set(genes)
    pylab.figure()
    matplotlib_venn.venn2([set(significant_genes_network1), set(significant_genes_network2)],
                          set_labels=[table1, table2])
    print "Overlap between significant genes (Fisher's exact test p-value): " + str(scipy.stats.fisher_exact(
        numpy.array([[len(set(significant_genes_network1) & set(significant_genes_network2)),
                      len(set(significant_genes_network1))
                      - len(set(significant_genes_network1) & set(significant_genes_network2))],
                     [len(set(significant_genes_network2))
                      - len(set(significant_genes_network1) & set(significant_genes_network2)),
                      len(genes) + len(set(significant_genes_network1) & set(significant_genes_network2))
                      - len(set(significant_genes_network1)) - len(set(significant_genes_network2))]]))[1])
    if is_rooted:
        pylab.figure()
        pylab.scatter(common_significant_genes_centroids_network1,
                      common_significant_genes_centroids_network2, alpha=0.2, s=8)
        x = numpy.linspace(int(min(common_significant_genes_centroids_network1
                                    + common_significant_genes_centroids_network2)),
                            int(max(common_significant_genes_centroids_network1
                                    + common_significant_genes_centroids_network2) + 1), 10)
        pylab.plot(x, x, 'k-')
        pylab.xlim(int(min(common_significant_genes_centroids_network1 + common_significant_genes_centroids_network2)),
                   int(max(common_significant_genes_centroids_network1 + common_significant_genes_centroids_network2))
                   + 1)
        pylab.ylim(int(min(common_significant_genes_centroids_network1 + common_significant_genes_centroids_network2)),
                   int(max(common_significant_genes_centroids_network1 + common_significant_genes_centroids_network2))
                   + 1)
        pylab.xlabel('Centroids ' + table1)
        pylab.ylabel('Centroids ' + table2)
        print "Pearson's correlation between centroids: " \
              + str(scipy.stats.pearsonr(common_significant_genes_centroids_network1,
                                         common_significant_genes_centroids_network2))
        pylab.figure()
        pylab.scatter(common_significant_genes_dispersions_network1, common_significant_genes_dispersions_network2,
                      alpha=0.2, s=8)
        x = numpy.linspace(int(min(common_significant_genes_dispersions_network1
                                   + common_significant_genes_dispersions_network2)),
                           int(max(common_significant_genes_dispersions_network1
                                   + common_significant_genes_dispersions_network2)) + 1, 10)
        pylab.plot(x, x, 'k-')
        pylab.xlim(int(min(common_significant_genes_dispersions_network1
                           + common_significant_genes_dispersions_network2)),
                   int(max(common_significant_genes_dispersions_network1
                           + common_significant_genes_dispersions_network2)) + 1)
        pylab.ylim(int(min(common_significant_genes_dispersions_network1
                           + common_significant_genes_dispersions_network2)),
                   int(max(common_significant_genes_dispersions_network1
                           + common_significant_genes_dispersions_network2)) + 1)
        pylab.xlabel('Dispersion ' + table1)
        pylab.ylabel('Dispersion ' + table2)
        print "Pearson's correlation between dispersions: " \
              + str(scipy.stats.pearsonr(common_significant_genes_dispersions_network1,
                                         common_significant_genes_dispersions_network2))
    pylab.show()


def hierarchical_clustering(mat, method='average', cluster_distance=True, labels=None, thres=0.65):
    """
    Performs hierarchical clustering based on distance matrix 'mat' using the method specified by 'method'.
    Optional argument 'labels' may specify a list of labels. If cluster_distance is True, the clustering is
    performed on the distance matrix using euclidean distance. Otherwise, mat specifies the distance matrix for
    clustering. Adapted from
    http://stackoverflow.com/questions/7664826/how-to-get-flat-clustering-corresponding-to-color-clusters-in-the-dendrogram-cre
    Not subjected to copyright.
    """
    distance_matrix = numpy.array(mat)
    if not cluster_distance:
        scipy_distance_matrix = scipy.spatial.distance.squareform(distance_matrix)
    else:
        scipy_distance_matrix = scipy.spatial.distance.pdist(distance_matrix, metric='euclidean')
    fig = pylab.figure(figsize=(8, 8))
    axes_left_dendrogram = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    linkage = sch.linkage(scipy_distance_matrix, method=method)
    left_dendrogram = sch.dendrogram(linkage, orientation='right', color_threshold=thres*max(linkage[:, 2]))
    axes_left_dendrogram.set_xticks([])
    axes_left_dendrogram.set_yticks([])
    axes_top_dendrogram = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    top_dendrogram = sch.dendrogram(linkage, color_threshold=thres*max(linkage[:, 2]))
    axes_top_dendrogram.set_xticks([])
    axes_top_dendrogram.set_yticks([])
    axes_heatmap = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    index_left = left_dendrogram['leaves']
    index_top = top_dendrogram['leaves']
    distance_matrix = distance_matrix[index_left, :]
    distance_matrix = distance_matrix[:, index_top]
    heatmap = axes_heatmap.matshow(distance_matrix, aspect='auto', origin='lower', cmap=pylab.get_cmap('jet_r'))
    if labels is None:
        axes_heatmap.set_xticks([])
        axes_heatmap.set_yticks([])
    else:
        axes_heatmap.set_xticks(range(len(labels)))
        labels_ordered = [labels[index_left[m]] for m in range(len(labels))]
        axes_heatmap.set_xticklabels(labels_ordered)
        axes_heatmap.set_yticks(range(len(labels)))
        axes_heatmap.set_yticklabels(labels_ordered)
        for tick in pylab.gca().xaxis.iter_ticks():
            tick[0].label2On = False
            tick[0].label1On = True
            tick[0].label1.set_rotation('vertical')
        for tick in pylab.gca().yaxis.iter_ticks():
            tick[0].label2On = True
            tick[0].label1On = False
    axes_colorbar = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    pylab.colorbar(heatmap, cax=axes_colorbar)
    pylab.show()
    return left_dendrogram


def find_clusters(z):
    clusters = {}
    for level_index, level in enumerate(z['dcoord']):
        for leaf_index, leaf in enumerate(level):
            if leaf == 0.0:
                if z['color_list'][level_index] not in clusters.keys():
                    clusters[z['color_list'][level_index]] = \
                        [z['leaves'][int((z['icoord'][level_index][leaf_index]-5.0)/10.0)]]
                else:
                    clusters[z['color_list'][level_index]].append(z['leaves'][int((z['icoord'][level_index][leaf_index]
                                                                                   -5.0)/10.0)])
    return clusters


"""
CLASSES
"""


class Preprocess(object):
    """
    Class for preparing and filtering data based on RNA spike in read counts. It takes as input one or more
    files with read counts produced by HTSeq-count. These can be all from a same time point or from multiple time
    points. It permits to read, filter and organize the data to put it in the appropriate form for SCTDA.
    """
    def __init__(self, files, timepoints, libs, cells, spike='_null_'):
        """
        Initializes the class by providing a list of files ('files'), timepoints ('timepoints') and library id's
        ('libs'), as well as the number of cells per file ('cells'), which can be a list, and the common identifier
        for the RNA spike-in reads ('spike'). The order of the genes must be the same in all files.
        """
        self.has_sigmoid_fit = False
        self.complexity = []
        self.sigmoid_fit_residuals = {}
        self.is_subsampled = False
        self.target_number_of_transcripts = 0.0
        self.transcript_counts_flatten = []
        self.spike_prefix = spike
        if type(cells) != list:
            self.number_of_cells_in_file = [cells] * len(files)
        else:
            self.number_of_cells_in_file = cells
        self.file_names = files
        self.long = list(numpy.repeat(timepoints, self.number_of_cells_in_file))
        self.batch = list(numpy.repeat(libs, self.number_of_cells_in_file))
        self.spike_transcript_counts_normalized_means_per_cell_flatten = []
        self.spike_counts_ratio_flatten = []
        self.spike_transcript_counts_normalized_means_per_cell_dict = {}
        spike_transcript_total_counts = {}
        self.transcript_total_counts = {}
        self.spike_counts_ratio = {}
        self.spike_transcript_counts_flatten = []
        transcript_counts = {}
        spike_transcript_counts = {}
        self.gene_names = []
        self.transcript_total_counts_flatten = []
        for file_name_index, file_name in enumerate(self.file_names):
            spike_transcript_counts_normalized = []
            spike_transcript_total_counts[file_name] = numpy.zeros(self.number_of_cells_in_file[file_name_index])
            self.transcript_total_counts[file_name] = numpy.zeros(self.number_of_cells_in_file[file_name_index])
            self.spike_counts_ratio[file_name] = numpy.zeros(self.number_of_cells_in_file[file_name_index])
            transcript_counts[file_name] = []
            spike_transcript_counts[file_name] = []
            file = open(file_name, 'r')
            spike_counter = 0
            for line in file:
                line_split = line[:-1].split('\t')
                transcript_counts_line = numpy.array(map(lambda x: float(x), line_split[1:]))
                if spike in line_split[0]:
                    spike_transcript_counts[file_name].append(transcript_counts_line)
                    spike_transcript_total_counts[file_name] += transcript_counts_line
                    if numpy.mean(transcript_counts_line) > 0.0:
                        spike_transcript_counts_normalized.append(transcript_counts_line/numpy.mean(
                            transcript_counts_line))
                    spike_counter += 1
                else:
                    self.transcript_total_counts[file_name] += transcript_counts_line
                    transcript_counts[file_name].append(transcript_counts_line)
                    if file_name_index == 0:
                        self.gene_names.append(line_split[0])
            if spike != '_null_':
                self.spike_counts_ratio[file_name] = spike_transcript_total_counts[file_name] / \
                                                     self.transcript_total_counts[file_name]
                self.spike_counts_ratio_flatten += list(self.spike_counts_ratio[file_name])
            file.close()
            spike_transcript_counts_normalized_means_per_cell = []
            for spike_transcript_counts_normalized_cell in \
                    numpy.transpose(numpy.array(spike_transcript_counts_normalized)):
                spike_transcript_counts_normalized_means_per_cell.append(numpy.mean(
                    spike_transcript_counts_normalized_cell))
            self.spike_transcript_counts_normalized_means_per_cell_flatten += \
                spike_transcript_counts_normalized_means_per_cell
            self.spike_transcript_counts_normalized_means_per_cell_dict[file_name] = \
                spike_transcript_counts_normalized_means_per_cell
            self.transcript_total_counts_flatten += list(self.transcript_total_counts[file_name])
            self.transcript_counts_flatten += list(numpy.transpose(numpy.array(transcript_counts[file_name])))
            self.spike_transcript_counts_flatten += list(numpy.transpose(numpy.array(spike_transcript_counts[file_name])))
        for gene_transcript_counts in self.transcript_counts_flatten:
            self.complexity.append(len(gene_transcript_counts) - list(gene_transcript_counts).count(0.0))
        self.transcript_total_counts_flatten = numpy.array(self.transcript_total_counts_flatten)
        self.transcript_counts_flatten = numpy.transpose(numpy.array(self.transcript_counts_flatten))
        self.spike_transcript_counts_flatten = numpy.transpose(numpy.array(self.spike_transcript_counts_flatten))
        self.which_genes = numpy.ones(len(self.transcript_counts_flatten), dtype=bool)
        self.which_cells = numpy.ones(len(self.transcript_counts_flatten[0]), dtype=bool)
        self.which_cells_subsampled = numpy.ones(len(self.transcript_counts_flatten[0]), dtype=bool)
        self.which_cells_backup = numpy.ones(len(self.transcript_counts_flatten[0]), dtype=bool)
        self.transcript_counts_flatten_subsampled = self.transcript_counts_flatten
        self.complexity_subsampled = self.complexity
        self.tpms = numpy.log2(1.0 + 1.0e6 * self.transcript_counts_flatten / self.transcript_total_counts_flatten)
        self.tpms = self.tpms[self.tpms > 0.0]
        self.gene_names = numpy.array(self.gene_names)

    def subsample(self, n):
        """
        Subsample counts so that each cell has exactly n total counts. If n = 0 sets the state of the Preprocess
        instance to the original unsampled situation. If a cell has less than n transcripts, it is removed from
        the analysis. Preprocess.which_samples indicates the cells that are kept in the analysis.
        """
        if n > 0:
            self.is_subsampled = True
            self.target_number_of_transcripts = float(n)
            self.which_cells_subsampled = (self.transcript_total_counts_flatten >= n)
            self.which_cells = self.which_cells_subsampled & self.which_cells_backup
            transcript_counts_normalized_flatten = self.transcript_counts_flatten / self.transcript_total_counts_flatten
            for cell_index, has_cell in enumerate(list(self.which_cells)):
                if not has_cell:
                    transcript_counts_normalized_flatten[cell_index] = 0.0
            self.transcript_counts_flatten_subsampled = []
            self.complexity_subsampled = []
            for cell_index in range(self.transcript_counts_flatten.shape[1]):
                if self.which_cells_subsampled[cell_index]:
                    self.transcript_counts_flatten_subsampled.append(
                        numpy.random.multinomial(n, transcript_counts_normalized_flatten[:, cell_index]))
                else:
                    self.transcript_counts_flatten_subsampled.append(
                        numpy.zeros(len(transcript_counts_normalized_flatten[:, cell_index])))
                last_cell_transcript_counts_flatten_subsampled = self.transcript_counts_flatten_subsampled[-1]
                self.complexity_subsampled.append(len(last_cell_transcript_counts_flatten_subsampled)
                                                  - list(last_cell_transcript_counts_flatten_subsampled).count(0.0))
            self.transcript_counts_flatten_subsampled = \
                numpy.transpose(numpy.array(self.transcript_counts_flatten_subsampled))
        else:
            self.is_subsampled = False
            self.transcript_counts_flatten_subsampled = self.transcript_counts_flatten
            self.complexity_subsampled = self.complexity
            self.target_number_of_transcripts = 0.0
            self.which_cells = self.which_cells_backup
            self.which_cells_subsampled = numpy.ones(len(self.transcript_counts_flatten[0]), dtype=bool)
        self.has_sigmoid_fit = False
        print str(int(list(self.which_cells).count(True))) + ' cells remain after subsmapling'

    def show_statistics(self):
        """
        Displays some basic statistics of the data. These are summarized in two plots. The first contains the ratio
        spike-in reads/uniquely mapped reads versus the ratio spike-in reads/average number of spike-in reads in the
        library. This plot is not displayed when there are not spike-ins in the analysis. The second is a
        histogram of the read counts expressed as transcripts per million (TPM), in a log_2(1+TPM) scale. The third
        plot shows the fraction of dropout events against the average expression for each gene. The four plot shows
        the distribution of the totl number of transcripts per cell. The fifth plot shows the distribution of
        complexity across cells. When subsampling and/or cell filtering have been performed, the corresponding
        plots for the subsampled data are overlaid in red.
        """
        if self.spike_prefix != '_null_':
            cell_color = []
            for gene_transcript_counts in list(self.which_cells):
                if gene_transcript_counts and list(self.which_cells).count(False) > 0:
                    cell_color.append('r.')
                else:
                    cell_color.append('b.')
            pylab.figure()
            for cell_index in range(len(self.spike_transcript_counts_normalized_means_per_cell_flatten)):
                pylab.plot(self.spike_transcript_counts_normalized_means_per_cell_flatten[cell_index],
                           self.spike_counts_ratio_flatten[cell_index], cell_color[cell_index], alpha=0.6)
            pylab.yscale('log')
            pylab.xlabel('Spike-in reads / average spike-in reads library')
            pylab.ylabel('Spike-in reads / uniquely mapped reads')
        pylab.figure()
        pylab.hist(self.tpms, 100, alpha=0.6, color='b')
        if self.is_subsampled:
            transcript_counts_flatten_masked = self.transcript_counts_flatten_subsampled[:, self.which_cells]
        else:
            transcript_counts_flatten_masked = self.transcript_counts_flatten[:, self.which_cells]
        if list(self.which_cells).count(False) > 0 or self.is_subsampled:
            if self.is_subsampled:
                tpms_masked = numpy.log2(1.0 + 1.0e6 * transcript_counts_flatten_masked / self.target_number_of_transcripts)
            else:
                tpms_masked = numpy.log2(1.0 + 1.0e6 * transcript_counts_flatten_masked /
                                self.transcript_total_counts_flatten[self.which_cells])
            tpms_masked = tpms_masked[tpms_masked > 0.0]
            pylab.hist(tpms_masked, 100, alpha=0.8, color='r')
        pylab.xlabel('log_2 (1 + TPM)')
        pylab.figure()
        x = []
        y = []
        for gene_transcript_counts in list(self.transcript_counts_flatten):
            x.append(float(list(gene_transcript_counts).count(0.0))/float(len(gene_transcript_counts)))
            y.append(numpy.mean(gene_transcript_counts))
        pylab.scatter(y, x, alpha=0.2, s=5, c='b')
        if list(self.which_cells).count(False) > 0 or self.is_subsampled:
            x_selected_cells = []
            y_selected_cells = []
            for gene_transcript_counts in list(self.transcript_counts_flatten_subsampled[:, self.which_cells]):
                x_selected_cells.append(float(list(gene_transcript_counts).count(0.0))/
                                        float(len(gene_transcript_counts)))
                y_selected_cells.append(numpy.mean(gene_transcript_counts))
            pylab.scatter(y_selected_cells, x_selected_cells, alpha=0.7, s=5, c='r')
        if self.spike_prefix != '_null_':
            x_spikes = []
            y_spikes = []
            for gene_transcript_counts in list(self.spike_transcript_counts_flatten):
                x_spikes.append(float(list(gene_transcript_counts).count(0.0))/float(len(gene_transcript_counts)))
                y_spikes.append(numpy.mean(gene_transcript_counts))
            pylab.scatter(y_spikes, x_spikes, alpha=0.7, s=10, c='y')
        pylab.xscale('log')
        pylab.ylim(-0.05, 1.05)
        pylab.xlim(min(numpy.array(y)[numpy.array(y) > 0]), max(y))
        pylab.xlabel('Average transcripts per cell')
        pylab.ylabel('Fraction of cells with non-detected expression')
        pylab.figure()
        pylab.hist(numpy.log10(self.transcript_total_counts_flatten[self.transcript_total_counts_flatten > 0.0]),
                   30, alpha=0.6, color='g')
        pylab.xlabel('Total number of transcripts in the cell')
        pylab.figure()
        pylab.hist(self.complexity, 30, alpha=0.6, color='y')
        if list(self.which_cells).count(False) > 0 or self.is_subsampled:
            pylab.hist(numpy.array(self.complexity_subsampled)[self.which_cells], 30, alpha=0.6, color='r')
        pylab.xlabel('Cell complexity')
        print 'Minimum number of transcripts per cell: ' + \
              str(int(min(self.transcript_total_counts_flatten[self.which_cells])))
        print 'Minimum cell complexity: ' + str(int(min(numpy.array(self.complexity)[self.which_cells])))
        if self.is_subsampled:
            print 'Minimum number of transcripts per cell (subsampled): ' + str(int(self.target_number_of_transcripts))
            print 'Minimum cell complexity (subsampled): ' + \
                  str(int(min(numpy.array(self.complexity_subsampled)[self.which_cells])))
        pylab.show()

    def fit_sigmoid(self, to_spikes=False):
        """
        Fits a sigmoid function to model the dependence of the dropout rate with the average expression and
        assigns a z-score to each gene by fitting the residuals with a normal distribution, where the standard
        deviation is estimated from the lower 16th percentile of the data. If subsampled has been performed through
        subsample, it fits the sigmoid to the subsampled data. If to_spikes is set to True, it uses spike in data
        to fit the sigmoid. If cells have been filtered out, it is taken into account.
        """
        def sigmoid(xt, x0, k):
            yt = 1 / (1 + numpy.exp(k*(xt-x0)))
            return yt
        if to_spikes and self.spike_prefix != '_null_':
            spike_transcript_counts_flatten_masked = \
                list(numpy.array(self.spike_transcript_counts_flatten)[:, numpy.array(self.which_cells)])
            if self.is_subsampled:
                spike_transcript_counts_flatten_subsampled_masked = \
                    list(numpy.array(self.transcript_counts_flatten_subsampled)[:, numpy.array(self.which_cells)])
            else:
                spike_transcript_counts_flatten_subsampled_masked = \
                    list(numpy.array(self.transcript_counts_flatten)[:, numpy.array(self.which_cells)])
        elif self.is_subsampled:
            spike_transcript_counts_flatten_masked = \
                list(numpy.array(self.transcript_counts_flatten_subsampled)[:, self.which_cells])
        else:
            spike_transcript_counts_flatten_masked = \
                numpy.array(self.transcript_counts_flatten)[:, numpy.array(self.which_cells)]
        pylab.figure()
        y = []
        x = []
        for gene_transcript_counts in list(spike_transcript_counts_flatten_masked):
            if numpy.mean(gene_transcript_counts) > 0.0:
                y.append(float(list(gene_transcript_counts).count(0.0))/float(len(gene_transcript_counts)))
                x.append(numpy.mean(gene_transcript_counts))
        log_x = numpy.log10(numpy.array(x))
        sigmoid_parameters, _ = scipy.optimize.curve_fit(sigmoid, log_x, numpy.array(y))
        continuous_x = numpy.linspace(int(min(log_x[log_x > -numpy.infty])), int(max(log_x))+1, 50)
        fitted_y = sigmoid(continuous_x, *sigmoid_parameters)
        if to_spikes and self.spike_prefix != '_null_':
            y_spikes = []
            x_spikes = []
            for gene_transcript_counts in list(spike_transcript_counts_flatten_subsampled_masked):
                y_spikes.append(float(list(gene_transcript_counts).count(0.0))/float(len(gene_transcript_counts)))
                x_spikes.append(numpy.mean(gene_transcript_counts))
            log_x_spikes = numpy.log10(numpy.array(x_spikes))
            pylab.scatter(log_x_spikes, y_spikes, alpha=0.2, s=5, c='b')
        else:
            pylab.scatter(log_x, y, alpha=0.2, s=5, c='b')
        if self.spike_prefix != '_null_' and not self.is_subsampled:
            y_selected = []
            x_selected = []
            for gene_transcript_counts in list(self.spike_transcript_counts_flatten[:, self.which_cells]):
                y_selected.append(float(list(gene_transcript_counts).count(0.0))/float(len(gene_transcript_counts)))
                x_selected.append(numpy.mean(gene_transcript_counts))
            log_x_selected = numpy.log10(numpy.array(x_selected))
            pylab.scatter(log_x_selected, y_selected, alpha=0.7, s=10, c='y')
        pylab.plot(continuous_x, fitted_y, 'r-')
        pylab.xlabel('Average transcripts per cell')
        pylab.ylabel('Fraction of cells with non-detected expression')
        pylab.ylim(-0.05, 1.05)
        pylab.figure()
        self.sigmoid_fit_residuals = {}
        if to_spikes and self.spike_prefix != '_null_':
            for pair_index, (x1, y1) in enumerate(zip(list(log_x_spikes), y_spikes)):
                self.sigmoid_fit_residuals[pair_index] = y1 - sigmoid(x1, *sigmoid_parameters)
        else:
            for pair_index, (x1, y1) in enumerate(zip(list(log_x), y)):
                self.sigmoid_fit_residuals[pair_index] = y1 - sigmoid(x1, *sigmoid_parameters)
        pylab.hist(self.sigmoid_fit_residuals.values(), 200, normed=True)
        x_continuous_residuals = numpy.linspace(min(self.sigmoid_fit_residuals.values()),
                                                max(self.sigmoid_fit_residuals.values()), 1000)
        width = numpy.median(self.sigmoid_fit_residuals.values()) \
                - numpy.percentile(self.sigmoid_fit_residuals.values(), 16)
        pylab.plot(x_continuous_residuals,
                   scipy.stats.norm.pdf(x_continuous_residuals,
                                        numpy.median(self.sigmoid_fit_residuals.values()), width), 'r-')
        pylab.xlabel('Residuals')
        for gene_transcript_counts in self.sigmoid_fit_residuals.keys():
            self.sigmoid_fit_residuals[gene_transcript_counts] /= width
        self.has_sigmoid_fit = True
        pylab.show()

    def select_genes(self, n_cells=0, avg_counts=0.0, min_z=-numpy.infty):
        """
        Selects a set of genes, based on various criteria, that will be used for building the topological
        representation. Selects for genes that are expressed in at least n_cells,
        that have at least avg_counts counts across all cells, and that have a minimum z-score of min_z with
        respect to the sigmoid fit. Needs to be run after fit_sigmoid if min_z is specified. If subsampling
        has been performed, it considers the subsampled dataset.
        """
        if min_z > -numpy.infty and not self.has_sigmoid_fit:
            print 'fit_sigmoid() needs to be run before selecting genes'
        else:
            if self.is_subsampled:
                transcript_counts_flatten_subsampled_masked = \
                    self.transcript_counts_flatten_subsampled[:, self.which_cells]
            else:
                transcript_counts_flatten_subsampled_masked = self.transcript_counts_flatten[:, self.which_cells]
            genes_color = []
            y = []
            x = []
            n = 0
            for gene_index, gene_transcript_counts_subsampled_masked in \
                    enumerate(list(transcript_counts_flatten_subsampled_masked)):
                if numpy.mean(gene_transcript_counts_subsampled_masked) > 0.0:
                    y.append(float(list(gene_transcript_counts_subsampled_masked).count(0.0))/
                             float(len(gene_transcript_counts_subsampled_masked)))
                    x.append(numpy.mean(gene_transcript_counts_subsampled_masked))
                    if len(gene_transcript_counts_subsampled_masked)-\
                            list(gene_transcript_counts_subsampled_masked).count(0.0) >= \
                            n_cells and numpy.mean(gene_transcript_counts_subsampled_masked) >= \
                            avg_counts and self.sigmoid_fit_residuals[n] >= min_z:
                        self.which_genes[gene_index] = True
                        genes_color.append('r')
                    else:
                        self.which_genes[gene_index] = False
                        genes_color.append('b')
                    n += 1
                else:
                    self.which_genes[gene_index] = False
            pylab.figure()
            log_x = numpy.log10(numpy.array(x))
            pylab.scatter(log_x, y, alpha=0.5, s=5, c=genes_color)
            if self.spike_prefix != '_null_':
                y_spikes = []
                x_spikes = []
                for gene_transcript_counts_subsampled_masked in \
                        list(self.spike_transcript_counts_flatten[:, self.which_cells]):
                    y_spikes.append(float(list(gene_transcript_counts_subsampled_masked).count(0.0))/
                                    float(len(gene_transcript_counts_subsampled_masked)))
                    x_spikes.append(numpy.mean(gene_transcript_counts_subsampled_masked))
                log_x_spikes = numpy.log10(numpy.array(x_spikes))
                pylab.scatter(log_x_spikes, y_spikes, alpha=0.7, s=10, c='y')
            pylab.xlabel('Average transcripts per cell')
            pylab.ylabel('Fraction of cells with non-detected expression')
            pylab.ylim(-0.05, 1.05)
            print str(int(genes_color.count('r'))) + " genes were selected"
            pylab.show()

    def reset_genes(self):
        """
        Includes all genes in the analysis.
        """
        self.which_genes = numpy.ones(len(self.transcript_counts_flatten), dtype=bool)

    def select_cells(self, min_transcripts=0.0, min_cdr=0.0, filterXlow=0.0, filterXhigh=1.0e8,
                     filterYlow=0.0, filterYhigh=1.0e8):
        """
        Selects a set of cells, based on various criteria, that will be used for subsequent analysis.
        Parameter 'min_transcripts' sets the minimum total number transcripts. Parameter 'least min_cdr'
        specifies the minimum cell complexity. Parameters 'filterXlow' and 'filterXhigh' set respectively
        lower and upper bounds in the ratio between spike-in reads and the average number of spike-in reads
        in the library. Parameters 'filterYlow' and 'filterYhigh' set respectively lower and upper bounds
        in the ratio between spike-in reads and uniquely mapped reads. Subsampling is not taken into account
        in the conditions.
        """
        if (filterXlow != 0.0 or filterXhigh != 1.0e8 or
                    filterYlow != 0.0 or filterYhigh != 1.0e8) and self.spike_prefix == '_null_':
            print 'No spike-ins selected'
        else:
            for cell_id, (cell_total_counts, cell_complexity) in \
                    enumerate(zip(list(self.transcript_total_counts_flatten), self.complexity)):
                if cell_total_counts >= min_transcripts and cell_complexity >= min_cdr:
                    self.which_cells[cell_id] = True & self.which_cells_subsampled[cell_id]
                    self.which_cells_backup[cell_id] = True
                else:
                    self.which_cells[cell_id] = False
                    self.which_cells_backup[cell_id] = False
            if self.spike_prefix != '_null_':
                cell_counter = 0
                for cell_id, f in enumerate(self.file_names):
                    for nok in range(self.number_of_cells_in_file[cell_id]):
                        if (filterXhigh > self.spike_transcript_counts_normalized_means_per_cell_dict[f][nok]
                                > filterXlow and filterYlow < self.spike_counts_ratio[f][nok] < filterYhigh):
                            self.which_cells[cell_counter] &= True
                            self.which_cells_backup[cell_counter] &= True
                        else:
                            self.which_cells[cell_counter] = False
                            self.which_cells_backup[cell_counter] = False
                        cell_counter += 1
                colors = []
                for cell_total_counts in self.which_cells:
                    if cell_total_counts:
                        colors.append('r.')
                    else:
                        colors.append('b.')
                pylab.figure()
                for cell_id in range(len(self.spike_transcript_counts_normalized_means_per_cell_flatten)):
                    pylab.plot(self.spike_transcript_counts_normalized_means_per_cell_flatten[cell_id],
                               self.spike_counts_ratio_flatten[cell_id], colors[cell_id], alpha=0.6)
                pylab.yscale('log')
                pylab.xlabel('spike-in reads / average spike-in reads library')
                pylab.ylabel('spike-in reads / uniquely mapped reads')
                pylab.show()
            print str(int(list(self.which_cells_backup).count(True))) + " cells were selected"
            if self.is_subsampled:
                print '(' + str(int(list(self.which_cells).count(True))) + " cells after subsampling)"

    def reset_cells(self):
        """
        Includes all cells in the analysis. If subsampling has been performed, it is taken into account.
        """
        self.which_cells = self.which_cells_subsampled
        self.which_cells_backup = numpy.ones(len(self.transcript_counts_flatten[0]), dtype=bool)

    def save(self, name):
        """
        Produces two tab separated files, called 'name.all.tsv' and 'name.mapper.tsv', where rows are the cells
        in self.which_samples. The first column of 'name.all.tsv' contains a unique identifier of the cell,
        the second column contains the sampling timepoint, the third column contains the library id and the
        remaining columns contain to log_2(1+TPM) expression values for each gene. If subsampling has been performed,
        the data refers to the subsampled data, whereas a third file called 'name.no_subsampling.tsv' is also
        created with the same format, containing non-subsampled data. 'name.mapper.tsv' contains log_2(1+TPM)
        expression values for each gene in self.which_genes.
        """
        file = open(name + '.all.tsv', 'w')
        line = 'ID\ttimepoint\tlib\t'
        if self.is_subsampled:
            data = self.transcript_counts_flatten_subsampled
        else:
            data = self.transcript_counts_flatten
        data_transposed = numpy.transpose(data)
        for gene in list(self.gene_names):
            line += gene + '\t'
        file.write(line[:-1] + '\n')
        for cell_id, is_cell_included in enumerate(list(self.which_cells)):
            if is_cell_included:
                line = 'D' + str(self.long[cell_id]) + '_' + self.batch[cell_id] + '_' + str(cell_id) + '\t' \
                       + str(self.long[cell_id]) + '\t' + self.batch[cell_id] + '\t'
                for transcript_count in list(data_transposed[cell_id]):
                    if self.is_subsampled:
                        total_transcript_count = self.target_number_of_transcripts
                    else:
                        total_transcript_count = self.transcript_total_counts_flatten[cell_id]
                    line += str(numpy.log2(1.0+1000000.0*transcript_count/float(total_transcript_count))) + '\t'
                file.write(line[:-1] + '\n')
        file.close()
        if self.is_subsampled:
            file = open(name + '.no_subsampling.tsv', 'w')
            line = 'ID\ttimepoint\tlib\t'
            for gene in list(self.gene_names):
                line += gene + '\t'
            file.write(line[:-1] + '\n')
            for cell_id, is_cell_included in enumerate(list(self.which_cells)):
                if is_cell_included:
                    line = 'D' + str(self.long[cell_id]) + '_' + self.batch[cell_id] + '_' + str(cell_id) + '\t' \
                           + str(self.long[cell_id]) + '\t' + self.batch[cell_id] + '\t'
                    for transcript_count in list(numpy.transpose(self.transcript_counts_flatten)[cell_id]):
                        if self.is_subsampled:
                            total_transcript_count = self.target_number_of_transcripts
                        else:
                            total_transcript_count = self.transcript_total_counts_flatten[cell_id]
                        line += str(numpy.log2(1.0+1000000.0*transcript_count/float(total_transcript_count))) + '\t'
                    file.write(line[:-1] + '\n')
            file.close()
        file = open(name + '.mapper.tsv', 'w')
        line = ''
        for is_cell_included in list(self.gene_names[self.which_genes]):
            line += is_cell_included + '\t'
        file.write(line[:-1] + '\n')
        for cell_id, is_cell_included in enumerate(list(self.which_cells)):
            if is_cell_included:
                line = ''
                for transcript_count in list(data_transposed[cell_id, self.which_genes]):
                    if self.is_subsampled:
                        total_transcript_count = self.target_number_of_transcripts
                    else:
                        total_transcript_count = self.transcript_total_counts_flatten[cell_id]
                    line += str(numpy.log2(1.0+1000000.0*transcript_count/float(total_transcript_count))) + '\t'
                file.write(line[:-1] + '\n')
        file.close()


class TopologicalRepresentation(object):
    """
    Class for building a topological representation of the data using SakMapper
    """
    def __init__(self, table, lens='mds', metric='correlation', precomputed=False, **kwargs):
        """
        Initializes the class by providing the mapper input table generated by Preprocess.save(). The parameter 'metric'
        specifies the metric distance to be used ('correlation', 'euclidean' or 'neighbor'). The parameter 'lens'
        specifies the dimensional reduction algorithm to be used ('mds' or 'pca'). The rest of the arguments are
        passed directly to sklearn.manifold.MDS or sklearn.decomposition.PCA. It plots the low-dimensional projection
        of the data.
        """
        self.data_frame = pandas.read_table(table + '.mapper.tsv')
        if lens == 'neighbor':
            self.lens_data_mds = sakmapper.apply_lens(self.data_frame, lens=lens, **kwargs)
        elif lens == 'mds':
            if precomputed:
                self.lens_data_mds = sakmapper.apply_lens(self.data_frame, lens=lens, metric=metric,
                                                          dissimilarity='precomputed', **kwargs)
            else:
                self.lens_data_mds = sakmapper.apply_lens(self.data_frame, lens=lens, metric=metric, **kwargs)
        else:
            self.lens_data_mds = sakmapper.apply_lens(self.data_frame, lens=lens, **kwargs)
        pylab.figure()
        pylab.scatter(numpy.array(self.lens_data_mds)[:, 0], numpy.array(self.lens_data_mds)[:, 1], s=10, alpha=0.7)
        pylab.show()

    def save(self, name, resolution, gain, equalize=True, cluster='agglomerative', statistics='db', max_K=5):
        """
        Generates a topological representation using the Mapper algorithm with resolution and gain specified by the
        parameters 'resolution' and 'gain'. When equalize is set to True, patches are chosen such that they
        contain the same number of points. The parameter 'cluster' specifies the clustering method ('agglomerative' or
        'kmeans'). The parameter 'statistics' specifies the criterion for choosing the optimal number of clusters
        ('db' for Davies-Bouildin index, or 'gap' for the gap statistic). The parameter 'max_K' specifies the maximum
        number of clusters to be considered within each patch. The topological representation is stored in the files
        'name.gexf' and 'name.json'. It returns a dictionary with the patches.
        """
        network, clusters, patches = sakmapper.mapper_graph(self.data_frame, lens_data=self.lens_data_mds,
                                                          resolution=resolution,
                                                          gain=gain, equalize=equalize, clust=cluster,
                                                          stat=statistics, max_K=max_K)
        clusters_dictionary = {}
        for cluster_id, cluster in enumerate(clusters):
            clusters_dictionary[str(cluster_id)] = map(lambda x: int(x), cluster)
        with open(name + '.json', 'wb') as json_file:
            json.dump(clusters_dictionary, json_file)
        networkx.write_gexf(network, name + '.gexf')
        return patches


class UnrootedGraph(object):
    """
    Main class for topological analysis of non-longitudinal single cell RNA-seq expression data.
    """
    def __init__(self, name, table, shift=None, log2=True, posgl=False, csv=False, groups=True):
        """
        Initializes the class by providing the the common name ('name') of .gexf and .json files produced by
        e.g. ParseAyasdiGraph() and the name of the file containing the filtered raw data ('table'), as produced by
        Preprocess.save(). Optional argument 'shift' can be an integer n specifying that the first n columns
        of the table should be ignored, or a list of columns that should only be considered. If optional argument
        'log2' is False, it is assumed that the filtered raw data is in units of TPM instead of log_2(1+TPM).
        When optional argument 'posgl' is False, a files name.posg and name.posgl are generated with the positions
        of the graph nodes for visualization. When 'posgl' is True, instead of generating new positions, the
        positions stored in files name.posg and name.posgl are used for visualization of the topological graph. If
        connected is False, all connected components of the network are displayed. When
        'csv' is True, the input table is in CSV format. When 'groups' is False, the class is initialized with an
        empty group dictionary (e.g. required when the topological representation has been generated through
        TopologicalRepresentation.save()).
        """
        self.name = name
        self.g = networkx.read_gexf(name + '.gexf')
        listii = [len(aa.nodes()) for aa in list(networkx.connected_component_subgraphs(self.g))]
        indexii = listii.index(numpy.max(listii))
        self.gl = list(networkx.connected_component_subgraphs(self.g))[indexii]
        self.pl = self.gl.nodes()
        self.adj = numpy.array(networkx.to_numpy_matrix(self.gl, nodelist=self.pl))
        self.log2 = log2
        self.cellID = []
        self.libs = []
        self.cdr = []
        if not posgl:
            try:
                from networkx.drawing.nx_agraph import graphviz_layout
                self.posgl = graphviz_layout(self.gl, 'sfdp', '-Goverlap=false -GK=0.1')
                self.posg = graphviz_layout(self.g, 'sfdp', '-Goverlap=false -GK=0.1')
            except:
                self.posgl = networkx.spring_layout(self.gl)
                self.posg = networkx.spring_layout(self.g)
            with open(name + '.posgl', 'w') as handler:
                pickle.dump(self.posgl, handler)
            with open(name + '.posg', 'w') as handler:
                pickle.dump(self.posg, handler)
        else:
            with open(name + '.posgl', 'r') as handler:
                self.posgl = pickle.load(handler)
            with open(name + '.posg', 'r') as handler:
                self.posg = pickle.load(handler)
        with open(name + '.json', 'r') as handler:
            self.dic = json.load(handler)
        if groups:
            with open(name + '.groups.json', 'r') as handler:
                self.dicgroups = json.load(handler)
        else:
            self.dicgroups = {}
        if csv:
            cx = ','
        else:
            cx = '\t'
        with open(table, 'r') as f:
            self.dicgenes = {}
            self.geneindex = {}
            for n, line in enumerate(f):
                sp2 = numpy.array(line[:-1].split(cx))
                if csv:
                    sp = [x.split('|')[0] for x in sp2]
                else:
                    sp = sp2
                if shift is None:
                    if n == 0:
                        poi = sp
                        if 'lib' in list(sp):
                            coyt = list(sp).index('lib')
                        else:
                            coyt = -1
                        if 'timepoint' in list(sp):
                            ttt = list(sp).index('timepoint')
                        else:
                            ttt = -1
                    else:
                        poi = []
                        self.cdr.append(0.0)
                        for n2, mji in enumerate(sp):
                            if is_number(mji):
                                poi.append(mji)
                                if float(mji) > 0.0 and n2 != 0 and n2 != coyt and n2 != ttt:
                                    self.cdr[-1] += 1.0
                            elif n2 == coyt:
                                self.libs.append(mji)
                                poi.append(0.0)
                            elif n2 == 0:
                                self.cellID.append(mji)
                                poi.append(0.0)
                            else:
                                poi.append(0.0)
                elif type(shift) == int:
                    poi = sp[shift:]
                else:
                    poi = list(numpy.array(sp)[shift])
                if n == 0:
                    for u, q in enumerate(poi):
                        self.dicgenes[q] = []
                        self.geneindex[u] = q
                else:
                    for u, q in enumerate(poi):
                        self.dicgenes[self.geneindex[u]].append(float(q))
        self.samples = []
        for i in self.pl:
            self.samples += self.dic[i]
        self.samples = numpy.array(list(set(self.samples)))
        self.cellID = numpy.array(self.cellID)
        self.cdr = numpy.array(self.cdr)/float(len(self.cdr))

    def get_gene(self, genin, ignore_log=False, con=True):
        """
        Returns a dictionary that asigns to each node id the average value of the column 'genin' in the raw table.
        'genin' can be also a list of columns, on which case the average of all columns. The output is normalized
        such that the sum over all nodes is equal to 1. It also provides as an output the normalization factor, to
        convert the dictionary to log_2(1+TPM) units (or TPM units). When 'ignore_log' is True it treat entries as
        being in natural scale, even if self.log2 is True (used internally). When 'con' is False, it uses all
        nodes, not only the ones in the first connected component of the topological representation (used internally).
        """
        if genin is not None and 'lib_' in genin[:4]:
            return self.count_gene('lib', genin[genin.index('_')+1:], con=con)
        elif genin is not None and 'ID_' in genin[:3]:
            return self.count_gene('ID', genin[genin.index('_')+1:], con=con)
        elif genin == '_CDR':
            genecolor = {}
            lista = []
            for i in self.dic.keys():
                if con:
                    if str(i) in self.pl:
                        genecolor[str(i)] = 0.0
                        lista.append(i)
                else:
                    genecolor[str(i)] = 0.0
                    lista.append(i)
            kis = range(len(self.cdr))
            for i in sorted(lista):
                pol = 0.0
                for j in self.dic[i]:
                        pol += float(self.cdr[kis[j]])
                pol /= float(len(self.dic[i]))
                genecolor[str(i)] += pol
            tol = sum(genecolor.values())
            if tol > 0.0:
                for ll in genecolor.keys():
                    genecolor[ll] = genecolor[ll]/tol
            return genecolor, tol
        else:
            if type(genin) != list:
                genin = [genin]
            genecolor = {}
            lista = []
            for i in self.dic.keys():
                if con:
                    if str(i) in self.pl:
                        genecolor[str(i)] = 0.0
                        lista.append(i)
                else:
                    genecolor[str(i)] = 0.0
                    lista.append(i)
            for mju in genin:
                if mju is None:
                    for i in sorted(lista):
                        genecolor[str(i)] = 0.0
                else:
                    geys = self.dicgenes[mju]
                    kis = range(len(geys))
                    for i in sorted(lista):
                        pol = 0.0
                        if self.log2 and not ignore_log:
                            for j in self.dic[i]:
                                    pol += (numpy.power(2, float(geys[kis[j]]))-1.0)
                            pol = numpy.log2(1.0+(pol/float(len(self.dic[i]))))
                        else:
                            for j in self.dic[i]:
                                    pol += float(geys[kis[j]])
                            pol /= float(len(self.dic[i]))
                        genecolor[str(i)] += pol
            tol = sum(genecolor.values())
            if tol > 0.0:
                for ll in genecolor.keys():
                    genecolor[ll] = genecolor[ll]/tol
            return genecolor, tol

    def connectivity_pvalue(self, genin, n=500):
        """
        Returns statistical significance of connectivity of gene specified by 'genin', using 'n' permutations.
        """
        if genin is not None:
            jk = len(self.pl)
            pm = numpy.zeros((jk, n), dtype='float32')
            llm = list(numpy.arange(numpy.max(self.samples)+1)[self.samples])
            koi = {k: u for u, k in enumerate(llm)}
            geys = numpy.tile(self.dicgenes[genin], (n, 1))[:, self.samples]
            map(numpy.random.shuffle, geys)
            tot = numpy.zeros(n)
            for k, i in enumerate(self.pl):
                pk = geys[:, numpy.array(map(lambda x: koi[x], self.dic[i]))]
                q = pk.shape[1]
                if self.log2:
                    t1 = numexpr.evaluate('sum(2**pk - 1, 1)')/q
                    pm[k, :] = numexpr.evaluate('log1p(t1)')/0.693147
                    tot += pm[k, :]
                else:
                    pm[k, :] = numexpr.evaluate('sum(pk, 1)')/q
                    tot += pm[k, :]
            pm = pm/tot
            conn = (float(jk)/float(jk-1))*numpy.einsum('ij,ij->j', numpy.dot(self.adj, pm), pm)
            return numpy.mean(conn > self.connectivity(genin))
        else:
            return 0.0

    def connectivity(self, genis, ind=1):
        """
        Returns the value of order 'ind' connectivity for the gene or list of genes specified by 'genis'.
        """
        dicgen = self.get_gene(genis)[0]
        ex = []
        for uu in self.pl:
            ex.append(dicgen[uu])
        ex = numpy.array(ex)
        cor = float(len(self.pl))/float(len(self.pl)-1)
        return cor*numpy.dot(numpy.transpose(ex),
                             numpy.dot(numpy.linalg.matrix_power(self.adj, ind), ex))

    def delta(self, genis, group=None):
        """
        Returns the mean, minimum and maximum expression values of the gene or list of genes specified
        by argument 'genin'. Optional argument 'group' allows to restrict this method to one of the
        node groups specified in the file name.groups.json.
        """
        per = []
        dicgen, tot = self.get_gene(genis)
        if group is not None:
            if type(group) == list:
                for k in group:
                    per += self.dicgroups[k]
                per = list(set(per))
            else:
                per = self.dicgroups[group]
            mi = [dicgen[str(node)] for node in per]
        else:
            mi = [dicgen[node] for node in self.pl]
        if numpy.mean(mi) > 0.0:
            return numpy.mean(mi)*tot, numpy.min(mi)*tot, numpy.max(mi)*tot
        else:
            return 0.0, 0.0, 0.0

    def expr(self, genis, group=None):
        """
        Returns the number of rows with non-zero expression of gene or list of genes specified by argument 'genin'.
        Optional argument 'group' allows to restrict this method to one of the node groups specified in the file
        name.groups.json.
        """
        per = []
        if group is not None:
            if type(group) == list:
                for k in group:
                    per += self.dicgroups[k]
                per = list(set(per))
            else:
                per = self.dicgroups[group]
            po = []
            for q in per:
                po += list(self.dic[str(q)])
            po = list(set(po))
            pi = sum(1 for i in numpy.array(self.dicgenes[genis])[po] if i > 0.0)
        else:
            pi = sum(1 for i in self.dicgenes[genis] if i > 0.0)
        return pi

    def save(self, n=500, filtercells=0, filterexp=0.0, annotation={}):
        """
        Computes UnrootedGraph.expr(), UnrootedGraph.delta(), UnrootedGraph.connectivity(),
        UnrootedGraph.connectivity_pvalue() and Benjamini-Holchberg adjusted q-values for all
        genes that are expressed in more than 'filtercells' cells and whose maximum expression
        value is above 'filterexp'. The optional argument 'annotation' allos to include a dictionary
        with lists of genes to be annotated in the table. The output is stored in a tab separated
        file called name.genes.txt.
        """
        pol = []
        with open(self.name + '.genes.tsv', 'w') as ggg:
            cul = 'Gene\tCells\tMean\tMin\tMax\tConnectivity\tp_value\tq-value (BH)\t'
            for m in sorted(annotation.keys()):
                cul += m + '\t'
            ggg.write(cul[:-1] + '\n')
            lp = sorted(self.dicgenes.keys())
            for gi in lp:
                if self.expr(gi) > filtercells and self.delta(gi)[2] > filterexp:
                    pol.append(self.connectivity_pvalue(gi, n=n))
            por = benjamini_hochberg(pol)
            mj = 0
            for gi in lp:
                po = self.expr(gi)
                m1, m2, m3 = self.delta(gi)
                if po > filtercells and m3 > filterexp:
                    cul = gi + '\t' + str(po) + '\t' + str(m1) + '\t' + str(m2) + '\t' + str(m3) + '\t' + \
                        str(self.connectivity(gi)) + '\t' + str(pol[mj]) + '\t' + str(por[mj]) + '\t'
                    for m in sorted(annotation.keys()):
                        if gi in annotation[m]:
                            cul += 'Y' + '\t'
                        else:
                            cul += 'N' + '\t'
                    ggg.write(cul[:-1] + '\n')
                    mj += 1
        centr = []
        disp = []
        centr2 = []
        disp2 = []
        f = open(self.name + '.genes.tsv', 'r')
        for n, line in enumerate(f):
            if n > 0:
                sp = line[:-1].split('\t')
                if float(sp[7]) <= 0.05:
                    centr.append(float(sp[1]))
                    disp.append(float(sp[5]))
                else:
                    centr2.append(float(sp[1]))
                    disp2.append(float(sp[5]))
        f.close()
        pylab.scatter(centr2, disp2, alpha=0.2, s=9, c='b')
        pylab.scatter(centr, disp, alpha=0.3, s=9, c='r')
        pylab.xlabel('Cells')
        pylab.ylabel('Connectivity')
        pylab.yscale('log')
        pylab.ylim(0.01, 1)
        pylab.xlim(0, max(centr+centr2))
        pylab.show()

    def JSD_matrix(self, lista, maximum_matrix_entries=5*10**7, verbose=False):
        """
        Returns the Jensen-Shannon distance matrix of the list of genes specified by 'lista'. If 'verbose' is set to
        True, it prints progress on the screen. 'maximum_matrix_entries' limits memory usage. If the largest matrix constructed
        as part of the calculation exceeds 'maximum_matrix_entries', the job is divided into multiple jobs, performed in series.
        The largest matrix has dimension (# nodes)*(# genes tested)^2.
        """
        ge = numpy.array([self.get_gene(genis)[0].values() for genis in lista])
        ge2 = numpy.copy(ge)
        ge2[ge2 == 0] = 1
        plogp = numpy.sum(ge2*numpy.log2(ge2), axis=1)
        plogptile = numpy.tile(plogp, (len(lista), 1))
        if len(ge)*len(ge[0])**2 <= maximum_matrix_entries:
            if verbose:
                print 'within limits: %s' % (len(ge)*len(ge[0])**2)
            ge_tile = numpy.tile(ge,(len(lista), 1, 1))
            ge_tile_T = numpy.transpose(ge_tile, [1, 0, 2])
            pq = 0.5*(ge_tile + ge_tile_T)
            pq[pq == 0] = 1
            pqlogpq = numpy.sum(pq*numpy.log2(pq), axis=2)
        else:
            group_number = len(ge)*len(ge[0])**2 / maximum_matrix_entries + 1
            group_length = len(ge) / group_number
            dd = range(0, len(ge), group_length)
            if dd[-1] != len(ge):
                dd += [len(ge)]
            if verbose:
                print 'outside limits: %s' % (len(ge)*len(ge[0])**2)
                print 'group_number = %s' % group_number
                print 'group_length = %s' % group_length
                print 'dd = %s'%dd
            for i in range(len(dd)-1):
                if verbose:
                    print 'i = %s' % i
                ge_tile = numpy.tile(ge, (dd[i+1] - dd[i], 1, 1))
                ge_tile_T = numpy.transpose(numpy.tile(ge[range(dd[i], dd[i+1])], (len(ge), 1, 1)), [1, 0, 2])
                pq = 0.5*(ge_tile + ge_tile_T)
                pq[pq == 0] = 1
                sliver = numpy.sum(pq*numpy.log2(pq), axis=2)
                if i == 0:
                    pqlogpq = sliver
                else:
                    pqlogpq = numpy.vstack((pqlogpq, sliver))
        jsdiv = 0.5 * (plogptile + plogptile.T - 2*pqlogpq)
        return numpy.sqrt(jsdiv)

    def cor_matrix(self, lista, c=1):
        """
        Returns correlation distance matrix of the list of genes specified by 'lista'. It uses 'c' cores for the
        computation.
        """
        geys = numpy.array([self.dicgenes[mju] for mju in lista])
        return sklearn.metrics.pairwise.pairwise_distances(geys, metric='correlation', n_jobs=c)

    def adjacency_matrix(self, lista, ind=1, verbose=False):
        """
        Returns the adjacency matrix of order 'ind' of the list of genes specified by 'lista'. If 'verbose' is set to
        True, it prints progress on the screen.
        """
        cor = float(len(self.pl))/float(len(self.pl)-1)
        pol = {}
        mat = []
        for genis in lista:
            pol[genis] = self.get_gene(genis)[0]
        for n, m1 in enumerate(lista):
            if verbose:
                print n
            ex1 = []
            for uu in self.pl:
                ex1.append(pol[m1][uu])
            ex1 = numpy.array(ex1)
            mat.append([])
            for m2 in lista:
                ex2 = []
                for uu in self.pl:
                    ex2.append(pol[m2][uu])
                ex2 = numpy.array(ex2)
                mat[-1].append(cor*numpy.dot(numpy.transpose(ex1),
                                             numpy.dot(numpy.linalg.matrix_power(self.adj, ind), ex2)))
        return numpy.array(mat)

    def draw(self, color, connected=True, labels=False, ccmap='jet', weight=8.0, save='', ignore_log=False,
             table=False, axis=[], a=0.4):
        """
        Displays topological representation of the data colored according to the expression of a gene, genes or
        list of genes, specified by argument 'color'. This can be a gene or a list of one, two or three genes or lists
        of genes, to be respectively mapped to red, green and blue channels. When only one gene or list of genes is
        specified, it uses color map specified by 'ccmap'. If optional argument 'connected' is set to True, only the
        largest connected component of the graph is displayed. If argument 'labels' is True, node id's are also
        displayed. Argument 'weight' allows to set a scaling factor for node sizes. When optional argument 'save'
        specifies a file name, the figure will be save in the file, in the format specified by its extension, and
        no plot will be displayed on the screen. When 'ignore_log' is True, it treat expression values as being in
        natural scale, even if self.log2 is True (used internally). When argument 'table' is True, it displays in
        addition a table with some statistics of the gene or genes. Optional argument 'axis' allows to specify axis
        limits in the form [xmin, xmax, ymin, ymax]. Parameter alpha specifies the alpha value of edges.
        """
        if connected:
            pg = self.gl
            pos = self.posgl
        else:
            pg = self.g
            pos = self.posg
        fig = pylab.figure()
        networkx.draw_networkx_edges(pg, pos, width=1, alpha=a)
        sizes = numpy.array([len(self.dic[node]) for node in pg.nodes()])*weight
        values = []
        if type(color) == str:
            color = [color]
        if type(color) == list and len(color) == 1:
            coloru, tol = self.get_gene(color[0], ignore_log=ignore_log, con=connected)
            values = [coloru[node] for node in pg.nodes()]
            nol = networkx.draw_networkx_nodes(pg, pos, node_color=values, node_size=sizes, cmap=pylab.get_cmap(ccmap))
            nol.set_edgecolor('k')
            polca = values
        elif type(color) == list and len(color) == 2:
            colorr, tolr = self.get_gene(color[0], ignore_log=ignore_log, con=connected)
            rmax = float(max(colorr.values()))
            if rmax == 0.0:
                rmax = 1.0
            colorb, tolb = self.get_gene(color[1], ignore_log=ignore_log, con=connected)
            bmax = float(max(colorb.values()))
            if bmax == 0.0:
                bmax = 1.0
            values = [(1.0-colorb[node]/bmax, max(1.0-(colorr[node]/rmax+colorb[node]/bmax), 0.0),
                       1.0-colorr[node]/rmax) for node in pg.nodes()]
            nol = networkx.draw_networkx_nodes(pg, pos, node_color=values, node_size=sizes)
            nol.set_edgecolor('k')
            polca = [(colorr[node], colorb[node]) for node in pg.nodes()]
        elif type(color) == list and len(color) == 3:
            colorr, tolr = self.get_gene(color[0], ignore_log=ignore_log, con=connected)
            rmax = float(max(colorr.values()))
            if rmax == 0.0:
                rmax = 1.0
            colorg, tolg = self.get_gene(color[1], ignore_log=ignore_log, con=connected)
            gmax = float(max(colorg.values()))
            if gmax == 0.0:
                gmax = 1.0
            colorb, tolb = self.get_gene(color[2], ignore_log=ignore_log, con=connected)
            bmax = float(max(colorb.values()))
            if bmax == 0.0:
                bmax = 1.0
            values = [(max(1.0-(colorg[node]/gmax+colorb[node]/bmax), 0.0),
                       max(1.0-(colorr[node]/rmax+colorb[node]/bmax), 0.0),
                       max(1.0-(colorr[node]/rmax+colorg[node]/gmax), 0.0)) for node in pg.nodes()]
            nol = networkx.draw_networkx_nodes(pg, pos, node_color=values, node_size=sizes)
            nol.set_edgecolor('k')
            polca = [(colorr[node], colorg[node], colorb[node]) for node in pg.nodes()]
        elif type(color) == list and len(color) == 4:
            colorr, tolr = self.get_gene(color[0], ignore_log=ignore_log, con=connected)
            rmax = float(max(colorr.values()))
            if rmax == 0.0:
                rmax = 1.0
            colorg, tolg = self.get_gene(color[1], ignore_log=ignore_log, con=connected)
            gmax = float(max(colorg.values()))
            if gmax == 0.0:
                gmax = 1.0
            colorb, tolb = self.get_gene(color[2], ignore_log=ignore_log, con=connected)
            bmax = float(max(colorb.values()))
            if bmax == 0.0:
                bmax = 1.0
            colord, told = self.get_gene(color[3], ignore_log=ignore_log, con=connected)
            dmax = float(max(colord.values()))
            if dmax == 0.0:
                dmax = 1.0
            values = [(max(1.0-(colorg[node]/gmax+colorb[node]/bmax), 0.0),
                       max(1.0-(colorr[node]/rmax+colorb[node]/bmax+0.36*colord[node]/dmax), 0.0),
                       max(1.0-(colorr[node]/rmax+colorg[node]/gmax+colord[node]/dmax), 0.0)) for node in pg.nodes()]
            nol = networkx.draw_networkx_nodes(pg, pos, node_color=values, node_size=sizes)
            nol.set_edgecolor('k')
            polca = [(colorr[node], colorg[node], colorb[node], colord[node]) for node in pg.nodes()]
        if labels:
            networkx.draw_networkx_labels(pg, pos, font_size=5, font_family='sans-serif')
        frame1 = pylab.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        if table:
            if type(color) == str:
                cell_text = [[str(min(values)*tol), str(max(values)*tol), str(numpy.median(values)*tol),
                              str(self.expr(color)), str(self.connectivity(color, ind=1)),
                              str(self.connectivity_pvalue(color, n=500))]]
                columns = ['Min.', 'Max.', 'Median', 'Cells', 'Connectivity', 'p value']
                rows = [color]
                pylab.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='bottom')
            elif type(color) == list and len(color) == 1:
                cell_text = [[str(min(values)*tol), str(max(values)*tol), str(numpy.median(values)*tol),
                              str(self.expr(color[0])),
                              str(self.connectivity(color[0], ind=1)),
                              str(self.connectivity_pvalue(color[0], n=500))]]
                columns = ['Min.', 'Max.', 'Median', 'Cells', 'Connectivity', 'p value']
                rows = [color[0]]
                pylab.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='bottom')
            elif type(color) == list and len(color) == 2:
                valuesr = colorr.values()
                valuesb = colorb.values()
                cell_text = [[str(min(valuesr)*tolr), str(max(valuesr)*tolr), str(numpy.median(valuesr)*tolr),
                              str(self.expr(color[0])),
                              str(self.connectivity(color[0], ind=1)),
                              str(self.connectivity_pvalue(color[0], n=500))], [str(min(valuesb)*tolb),
                              str(max(valuesb)*tolb), str(numpy.median(valuesb)*tolb),
                              str(self.expr(color[1])),
                              str(self.connectivity(color[1], ind=1)),
                              str(self.connectivity_pvalue(color[1], n=500))]]
                columns = ['Min.', 'Max.', 'Median', 'Cells', 'Connectivity', 'p value']
                rows = [color[0], color[1]]
                pylab.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='bottom', rowColours=['r', 'b'])
            elif type(color) == list and len(color) == 3:
                valuesr = colorr.values()
                valuesg = colorg.values()
                valuesb = colorb.values()
                cell_text = [[str(min(valuesr)*tolr), str(max(valuesr)*tolr), str(numpy.median(valuesr)*tolr),
                              str(self.expr(color[0])),
                              str(self.connectivity(color[0], ind=1)),
                              str(self.connectivity_pvalue(color[0], n=500))],
                             [str(min(valuesg)*tolg), str(max(valuesg)*tolg), str(numpy.median(valuesg)*tolg),
                              str(self.expr(color[1])),
                              str(self.connectivity(color[1], ind=1)),
                              str(self.connectivity_pvalue(color[1], n=500))],
                             [str(min(valuesb)*tolb), str(max(valuesb)*tolb), str(numpy.median(valuesb)*tolb),
                              str(self.expr(color[2])),
                              str(self.connectivity(color[2], ind=1)),
                              str(self.connectivity_pvalue(color[2], n=500))]]
                columns = ['Min.', 'Max.', 'Median', 'Cells', 'Connectivity', 'p value']
                rows = [color[0], color[1], color[2]]
                the_table = pylab.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='bottom',
                                        rowColours=['r', 'g', 'b'], colWidths=[0.08] * 7)
                the_table.scale(1.787, 1)
                pylab.subplots_adjust(bottom=0.2)
        if len(axis) == 4:
            pylab.axis(axis)
        if save == '':
            pylab.show()
        else:
            fig.savefig(save)
        return polca

    def count_gene(self, genin, cond, con=True):
        """
        Returns a dictionary that assigns to each node id the fraction of cells in the node for which column 'genin'
        is equal to 'cond'. When optional argument 'con' is False, it uses all nodes, not only the ones in the first
        connected component of the topological representation (used internally).
        """
        genecolor = {}
        lista = []
        for i in self.dic.keys():
            if con:
                if str(i) in self.pl:
                    genecolor[str(i)] = 0.0
                    lista.append(i)
            else:
                genecolor[str(i)] = 0.0
                lista.append(i)
        if genin is None:
            for i in sorted(lista):
                genecolor[str(i)] = 0.0
        else:
            if genin == 'lib':
                geys = self.libs
            elif genin == 'ID':
                geys = list(self.cellID)
            else:
                geys = self.dicgenes[genin]
            for i in sorted(lista):
                pol = 0.0
                for j in self.dic[i]:
                    if geys[j] == cond:
                        pol += 1.0
                genecolor[str(i)] = pol/float(len(self.dic[i]))
        tol = sum(genecolor.values())
        if tol > 0.0:
            for ll in genecolor.keys():
                genecolor[ll] = genecolor[ll]/tol
        return genecolor, tol

    def show_statistics(self):
        """
        Shows several statistics of the data. The first plot shows the distribution of the number of cells per node
        in the topological representation. The second plot shows the distribution of the number of common cells
        between nodes that share an edge in the topological representation. The third plot contains the distribution
        of the number of nodes that contain the same cell. Finally, the fourth plot shows the distribution of
        transcripts in log_2(1+TPM) scale, after filtering.
        """
        x = map(len, self.dic.values())
        pylab.figure()
        pylab.hist(x, max(x)-1, alpha=0.6, color='b')
        pylab.xlabel('Cells per node')
        x = []
        for q in self.g.edges():
            x.append(len(set(self.dic[q[0]]).intersection(self.dic[q[1]])))
        pylab.figure()
        pylab.hist(x, max(x)-1, alpha=0.6, color='g')
        pylab.xlabel('Shared cells between connected nodes')
        pel = []
        for m in self.dic.values():
            pel += list(m)
        q = []
        for m in range(max(pel)+1):
            o = pel.count(m)
            if o > 0:
                q.append(o)
        pylab.figure()
        pylab.hist(q, max(q)-1, alpha=0.6, color='r')
        pylab.xlabel('Number of nodes containing the same cell')
        pylab.figure()
        r = []
        for m in self.dicgenes.keys():
            r += list(self.dicgenes[m])
        r = [k for k in r if 30 > k > 0.0]
        pylab.hist(r, 100, alpha=0.6)
        pylab.xlabel('Expression')
        pylab.show()

    def cellular_subpopulations(self, threshold=0.05, min_cells=5, clus_thres=0.65):
        """
        Identifies potential transient cellular subpopulations. The parameter
        'threshold' sets an upper bound of the q-value of the genes that are considered in the analysis.
        The parameter 'min_cells' sets the minimum number of cells on which each of the genes considered in the
        analysis is expressed. Cellular subpopulations are determined by clustering the Jensen-Shannon distance
        matrix of the genes that pass all the constraints. The number of clusters is controlled in this case by
        the parameter 'clus_thres'. In both cases a list with the genes associated to each cluster is returned.
        It requires the presence of the file 'name.genes.tsv', produced by the method RotedGraph.save().
        """
        con = []
        dis = []
        nam = []
        f = open(self.name + '.genes.tsv', 'r')
        for n, line in enumerate(f):
            if n > 0:
                sp = line[:-1].split('\t')
                if float(sp[7]) < threshold and float(sp[1]) > min_cells:
                    nam.append(sp[0])
        f.close()
        mat2 = self.JSD_matrix(nam)
        return [map(lambda xx: nam[xx], m)
                for m in find_clusters(hierarchical_clustering(mat2, labels=nam,
                                                               cluster_distance=True, thres=clus_thres)).values()]


class RootedGraph(UnrootedGraph):
    """
    Inherits from UnrootedGraph. Main class for topological analysis of longitudinal single cell RNA-seq
    expression data.
    """
    def get_distroot(self, root):
        """
        Returns a dictionary of graph distances to node specified by argument 'root'
        """
        distroot = {}
        for i in sorted(self.pl):
            distroot[str(i)] = networkx.shortest_path_length(self.gl, str(root), i)
        return distroot

    def get_dendrite(self):
        """
        Returns function that for each graph node takes the value of the correlation between the graph distance function
        to the node and the sampling time fuunction specified by self.rootlane (used internally).
        """
        dendrite = {}
        daycolor = self.get_gene(self.rootlane, ignore_log=True)[0]
        for i in self.pl:
            distroot = self.get_distroot(i)
            x = []
            y = []
            for q in distroot.keys():
                if distroot[q] != max(distroot.values()):
                    x.append(daycolor[q]-min(daycolor.values()))
                    y.append(distroot[q])
            dendrite[str(i)] = -scipy.stats.spearmanr(x, y)[0]
        return dendrite

    def find_root(self, dendritem):
        """
        Given the output of RootedGraph.get_dendrite() as 'dendritem', it returns the less and the most differentiated
        nodes (used internally).
        """
        q = 1000.0
        q2 = -1000.0
        ind = 0
        ind2 = 0
        for n in dendritem.keys():
            if -2.0 < dendritem[n] < q:
                q = dendritem[n]
                ind = n
            if dendritem[n] > q2 and dendritem[n] > -2.0:
                q2 = dendritem[n]
                ind2 = n
        return ind, ind2

    def dendritic_graph(self):
        """
        Builds skeleton of the topological representation (used internally)
        """
        diam = networkx.diameter(self.gl)
        g3 = networkx.Graph()
        dicdend = {}
        for n in range(diam-1):
            nodedist = []
            for k in self.pl:
                dil = networkx.shortest_path_length(self.gl, self.root, k)
                if dil == n:
                    nodedist.append(str(k))
            g2 = self.gl.subgraph(nodedist)
            dicdend[n] = sorted(networkx.connected_components(g2))
            for n2, yu in enumerate(dicdend[n]):
                g3.add_node(str(n) + '_' + str(n2))
                if n > 0:
                    for n3, yu2 in enumerate(dicdend[n-1]):
                        if networkx.is_connected(self.gl.subgraph(list(yu)+list(yu2))):
                            g3.add_edge(str(n) + '_' + str(n2), str(n-1) + '_' + str(n3))
        return g3, dicdend

    def __init__(self, name, table, rootlane='timepoint', shift=None, log2=True, posgl=False, csv=False, groups=True):
        """
        Initializes the class by providing the the common name ('name') of .gexf and .json files produced by
        e.g. ParseAyasdiGraph(), the name of the file containing the filtered raw data ('table'), as produced by
        Preprocess.save(), and the name of the column that contains sampling time points. Optional argument
        'shift' can be an integer n specifying that the first n columns of the table should be ignored, or a
        list of columns that should only be considered. If optional argument 'log2' is False, it is assumed that
        the filtered raw data is in units of TPM instead of log_2(1+TPM). When optional argument 'posgl' is False,
        a files name.posg and name.posgl are generated with the positions of the graph nodes for visualization.
        When 'posgl' is True, instead of generating new positions, the positions stored in files name.posg and
        name.posgl are used for visualization of the topological graph.
        """
        UnrootedGraph.__init__(self, name, table, shift, log2, posgl, csv, groups)
        self.rootlane = rootlane
        self.root, self.leaf = self.find_root(self.get_dendrite())
        self.g3, self.dicdend = self.dendritic_graph()
        self.edgesize = []
        self.dicedgesize = {}
        self.edgesizeprun = []
        self.nodesize = []
        self.dicmelisa = {}
        self.nodesizeprun = []
        self.dicmelisaprun = {}
        for ee in self.g3.edges():
            yu = self.dicdend[int(ee[0].split('_')[0])][int(ee[0].split('_')[1])]
            yu2 = self.dicdend[int(ee[1].split('_')[0])][int(ee[1].split('_')[1])]
            self.edgesize.append(self.gl.subgraph(list(yu)+list(yu2)).number_of_edges()-self.gl.subgraph(yu).number_of_edges()
                                 - self.gl.subgraph(yu2).number_of_edges())
            self.dicedgesize[ee] = self.edgesize[-1]
        for ee in self.g3.nodes():
            lisa = []
            for uu in self.dicdend[int(ee.split('_')[0])][int(ee.split('_')[1])]:
                lisa += self.dic[uu]
            self.nodesize.append(len(set(lisa)))
            self.dicmelisa[ee] = set(lisa)
        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            self.posg3 = graphviz_layout(self.g3, 'sfdp')
        except:
            self.posg3 = networkx.spring_layout(self.g3)
        self.dicdis = self.get_distroot(self.root)
        pel2, tol = self.get_gene(self.rootlane, ignore_log=True)
        self.pel = numpy.array([pel2[m] for m in self.pl])*tol
        dr2 = self.get_distroot(self.root)
        self.dr = numpy.array([dr2[m] for m in self.pl])
        self.po = scipy.stats.linregress(self.pel, self.dr)

    def select_diff_path(self):
        """
        Returns a linear subgraph of the skeleton of the topological representation that maximizes the number of edges
        (used internally).
        """
        lista = []
        last = '0_0'
        while True:
            siz = 0
            novel = None
            for ee in self.dicedgesize.keys():
                if ((ee[0] == last and float(ee[1].split('_')[0]) > float(ee[0].split('_')[0]))
                    or (ee[1] == last and float(ee[0].split('_')[0]) > float(ee[1].split('_')[0]))) \
                        and self.dicedgesize[ee] > siz and ee not in lista:
                    novel = ee
                    siz = self.dicedgesize[ee]
            if novel is not None:
                lista.append(novel)
                if float(novel[1].split('_')[0]) > float(novel[0].split('_')[0]):
                    last = novel[1]
                else:
                    last = novel[0]
            else:
                break
        return lista

    def draw_skeleton(self, color, labels=False, ccmap='jet', weight=8.0, save='', ignore_log=False, markpath=False):
        """
        Displays skeleton of topological representation of the data colored according to the expression of a gene,
        genes or list of genes, specified by argument 'color'. This can be a gene or a list of one, two or three
        genes or lists of genes, to be respectively mapped to red, green and blue channels. When only one gene or
        list of genes is specified, it uses color map specified by 'ccmap'. If argument 'labels' is True, node id's
        are also displayed. Argument 'weight' allows to set a scaling factor for node sizes. When optional argument
        'save' specifies a file name, the figure will be save in the file, in the format specified by its extension, and
        no plot will be displayed on the screen. When 'ignore_log' is True, it treat expression values as being in
        natural scale, even if self.log2 is True (used internally). When argument 'markpath' is True, it highlights the
        linear path produced by RootedGraph.select_diff_path().
        """
        values = []
        pg = self.g3
        pos = self.posg3
        edgesize = self.edgesize
        nodesize = self.nodesize
        fig = pylab.figure()
        networkx.draw_networkx_edges(pg, pos,
                                     width=numpy.log2(numpy.array(edgesize)+1)*8.0/float(numpy.log2(1+max(edgesize))),
                                     alpha=0.6)
        if markpath:
            culer = self.select_diff_path()
            edgesize2 = [self.dicedgesize[m] for m in culer]
            networkx.draw_networkx_edges(pg, pos, edgelist=culer, edge_color='r',
                                         width=numpy.log2(numpy.array(edgesize2)+1)*8.0/float(numpy.log2(1+max(edgesize))),
                                         alpha=0.6)
        if type(color) == str or (type(color) == list and len(color) == 1):
            values = []
            for _ in pg.nodes():
                values.append(0.0)
            if type(color) == str:
                color = [[color]]
            for colorm in color[0]:
                geys = self.dicgenes[colorm]
                for llp, ee in enumerate(pg.nodes()):
                    pol = 0.0
                    if self.log2 and not ignore_log:
                        for uni in self.dicmelisa[ee]:
                                pol += (numpy.power(2, float(geys[uni]))-1.0)
                        pol = numpy.log2(1.0+(pol/float(len(self.dicmelisa[ee]))))
                    else:
                        for uni in self.dicmelisa[ee]:
                            pol += geys[uni]
                        pol /= len(self.dicmelisa[ee])
                    values[llp] += pol
            nol = networkx.draw_networkx_nodes(pg, pos, node_color=values,
                                         node_size=numpy.array(nodesize)*weight*50.0/float(max(nodesize)),
                                         cmap=pylab.get_cmap(ccmap))
            nol.set_edgecolor('k')
        elif type(color) == list and len(color) == 2:
            geysr = self.dicgenes[color[0]]
            geysb = self.dicgenes[color[1]]
            colorr = {}
            colorb = {}
            for ee in pg.nodes():
                polr = 0.0
                polb = 0.0
                if self.log2 and not ignore_log:
                    for uni in self.dicmelisa[ee]:
                            polr += (numpy.power(2, float(geysr[uni]))-1.0)
                            polb += (numpy.power(2, float(geysb[uni]))-1.0)
                    polr = numpy.log2(1.0+(polr/float(len(self.dicmelisa[ee]))))
                    polb = numpy.log2(1.0+(polb/float(len(self.dicmelisa[ee]))))
                else:
                    for uni in self.dicmelisa[ee]:
                        polr += geysr[uni]
                        polb += geysb[uni]
                    polr /= len(self.dicmelisa[ee])
                    polb /= len(self.dicmelisa[ee])
                colorr[ee] = polr
                colorb[ee] = polb
            rmax = float(max(colorr.values()))
            bmax = float(max(colorb.values()))
            values = [(1.0-colorb[node]/bmax, max(1.0-(colorr[node]/rmax+colorb[node]/bmax), 0.0),
                       1.0-colorr[node]/rmax) for node in pg.nodes()]
            nol = networkx.draw_networkx_nodes(pg, pos, node_color=values,
                                         node_size=numpy.array(nodesize)*weight*50.0/float(max(nodesize)))
            nol.set_edgecolor('k')
        elif type(color) == list and len(color) == 3:
            geysr = self.dicgenes[color[0]]
            geysg = self.dicgenes[color[1]]
            geysb = self.dicgenes[color[2]]
            colorr = {}
            colorg = {}
            colorb = {}
            for ee in pg.nodes():
                polr = 0.0
                polg = 0.0
                polb = 0.0
                if self.log2 and not ignore_log:
                    for uni in self.dicmelisa[ee]:
                            polr += (numpy.power(2, float(geysr[uni]))-1.0)
                            polg += (numpy.power(2, float(geysg[uni]))-1.0)
                            polb += (numpy.power(2, float(geysb[uni]))-1.0)
                    polr = numpy.log2(1.0+(polr/float(len(self.dicmelisa[ee]))))
                    polg = numpy.log2(1.0+(polg/float(len(self.dicmelisa[ee]))))
                    polb = numpy.log2(1.0+(polb/float(len(self.dicmelisa[ee]))))
                else:
                    for uni in self.dicmelisa[ee]:
                        polr += geysr[uni]
                        polg += geysg[uni]
                        polb += geysb[uni]
                    polr /= len(self.dicmelisa[ee])
                    polg /= len(self.dicmelisa[ee])
                    polb /= len(self.dicmelisa[ee])
                colorr[ee] = polr
                colorg[ee] = polg
                colorb[ee] = polb
            rmax = float(max(colorr.values()))
            gmax = float(max(colorg.values()))
            bmax = float(max(colorb.values()))
            values = [(max(1.0-(colorg[node]/gmax+colorb[node]/bmax), 0.0),
                       max(1.0-(colorr[node]/rmax+colorb[node]/bmax), 0.0),
                       max(1.0-(colorr[node]/rmax+colorg[node]/gmax), 0.0)) for node in pg.nodes()]
            nol = networkx.draw_networkx_nodes(pg, pos, node_color=values,
                                         node_size=numpy.array(nodesize)*weight*50.0/float(max(nodesize)))
            nol.set_edgecolor('k')
        frame1 = pylab.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        if labels:
            networkx.draw_networkx_labels(pg, pos, font_size=5, font_family='sans-serif')
        if save == '':
            pylab.show()
        else:
            fig.savefig(save)
        return values

    def centroid(self, genin, ignore_log=False):
        """
        Returns the centroid and dispersion of a genes or list of genes specified by argument 'genin'.
        When 'ignore_log' is True, it treat expression values as being in natural scale, even if self.log2 is
        True (used internally).
        """
        dicge = self.get_gene(genin, ignore_log=ignore_log)[0]
        pel1 = 0.0
        pel2 = 0.0
        pel3 = 0.0
        for node in self.pl:
            pel1 += self.dicdis[node]*dicge[node]
            pel2 += dicge[node]
        if pel2 > 0.0:
            cen = float(pel1)/float(pel2)
            for node in self.pl:
                pel3 += numpy.power(self.dicdis[node]-cen, 2)*dicge[node]
            return [(cen-self.po[1])/self.po[0], (numpy.sqrt(pel3/float(pel2))-self.po[1])/self.po[0]]
        else:
            return [None, None]

    def get_gene(self, genin, ignore_log=False, con=True):
        """
        Returns a dictionary that asigns to each node id the average value of the column 'genin' in the raw table.
        'genin' can be also a list of columns, on which case the average of all columns. The output is normalized
        such that the sum over all nodes is equal to 1. It also provides as an output the normalization factor, to
        convert the dictionary to log_2(1+TPM) units (or TPM units). When 'ignore_log' is True it treat entries as
        being in natural scale, even if self.log2 is True (used internally). When 'con' is False, it uses all
        nodes, not only the ones in the first connected component of the topological representation (used internally).
        Argument 'genin' may also be equal to the special keyword '_dist_root', on which case it returns the graph
        distance funtion to the root node. It can be also equal to 'timepoint_xxx', on which case it returns a
        dictionary with the fraction of cells belonging to timepoint xxx in each node.
        """
        if genin == '_dist_root':
            return self.get_distroot(self.root), 1.0
        elif genin is not None and 'timepoint_' in genin:
            return self.count_gene(self.rootlane, float(genin[genin.index('_')+1:]))
        else:
            return UnrootedGraph.get_gene(self, genin, ignore_log, con)

    def draw_expr_timeline(self, genin, ignore_log=False, path=False, save='', axis=[], smooth=False):
        """
        It displays the expression of a gene or list of genes, specified by argument 'genin', at different time points,
        as inferred from the distance to root function. When 'ignore_log' is True, it treat expression values as being
        in natural scale, even if self.log2 is True (used internally). When optional argument 'save' specifies a file
        name, the figure will be save in the file, in the format specified by its extension, and no plot will be
        displayed on the screen. Optional argument 'axis' allows to specify axis limits in the form
        [xmin, xmax, ymin, ymax]. If argument 'path' is True, expression is computed only across the linear path
        produced by RootedGraph.select_diff_path().
        """
        distroot_inv = {}
        if not path:
            pel = self.get_distroot(self.root)
            for m in pel.keys():
                if pel[m] not in distroot_inv.keys():
                    distroot_inv[pel[m]] = [m]
                else:
                    distroot_inv[pel[m]].append(m)
        else:
            pel = self.select_diff_path()
            cali = []
            for mmn in pel:
                cali.append(mmn[0])
                cali.append(mmn[1])
            cali = list(set(cali))
            for mmn in cali:
                distroot_inv[int(mmn.split('_')[0])] = self.dicdend[int(mmn.split('_')[0])][int(mmn.split('_')[1])]
        if type(genin) != list:
            genin = [genin]
        polter = {}
        for qsd in distroot_inv.keys():
            genecolor = {}
            lista = []
            for i in self.dic.keys():
                if str(i) in distroot_inv[qsd]:
                    genecolor[str(i)] = 0.0
                    lista += list(self.dic[i])
            pol = []
            for mju in genin:
                geys = self.dicgenes[mju]
                for j in lista:
                    if self.log2 and not ignore_log:
                        pol.append(numpy.power(2, float(geys[j]))-1.0)
                    else:
                        pol.append(float(geys[j]))
            pol = map(lambda xcv: numpy.log2(1+xcv), pol)
            polter[qsd] = [numpy.mean(pol)-numpy.std(pol), numpy.mean(pol), numpy.mean(pol)+numpy.std(pol)]
        x = []
        y = []
        y1 = []
        y2 = []
        for m in sorted(polter.keys()):
            x.append((m-self.po[1])/self.po[0])
            y1.append(polter[m][0])
            y.append(polter[m][1])
            y2.append(polter[m][2])
        xnew = numpy.linspace(min(x), max(x), 300)
        ynew = scipy.interpolate.spline(x, y, xnew)
        if smooth:
            ynew = scipy.signal.savgol_filter(ynew, 30, 3)
        fig = pylab.figure(figsize=(12, 3))
        pylab.fill_between(xnew, 0, ynew, alpha=0.5)
        pylab.ylim(0.0, max(ynew)*1.2)
        pylab.xlabel(self.rootlane)
        pylab.ylabel('<log2 (1+x)>')
        if len(axis) == 2:
            pylab.xlim(axis)
        elif len(axis) == 4:
            pylab.xlim(axis[0], axis[1])
            pylab.ylim(axis[2], axis[3])
        else:
            pylab.xlim(min(xnew), max(xnew))
        if save == '':
            pylab.show()
        else:
            fig.savefig(save)
        return polter

    def plot_CDR_correlation(self, doplot=True):
        """
        Displays correlation between sampling time points and CDR. It returns the two
        parameters of the linear fit, Pearson's r, p-value and standard error. If optional argument 'doplot' is
        False, the plot is not displayed.
        """
        pel2, tol = self.get_gene(self.rootlane, ignore_log=True)
        pel = numpy.array([pel2[m] for m in self.pl])*tol
        dr2 = self.get_gene('_CDR')[0]
        dr = numpy.array([dr2[m] for m in self.pl])
        po = scipy.stats.linregress(pel, dr)
        if doplot:
            pylab.scatter(pel, dr, s=9.0, alpha=0.7, c='r')
            pylab.xlim(min(pel), max(pel))
            pylab.ylim(0, max(dr)*1.1)
            pylab.xlabel(self.rootlane)
            pylab.ylabel('CDR')
            xk = pylab.linspace(min(pel), max(pel), 50)
            pylab.plot(xk, po[1]+po[0]*xk, 'k--', linewidth=2.0)
            pylab.show()
        return po

    def plot_rootlane_correlation(self):
        """
        Displays correlation between sampling time points and graph distance to root node. It returns the two
        parameters of the linear fit, Pearson's r, p-value and standard error.
        """
        pylab.scatter(self.pel, self.dr, s=9.0, alpha=0.7, c='r')
        pylab.xlim(min(self.pel), max(self.pel))
        pylab.ylim(0, max(self.dr)+1)
        pylab.xlabel(self.rootlane)
        pylab.ylabel('Distance to root node')
        xk = pylab.linspace(min(self.pel), max(self.pel), 50)
        pylab.plot(xk, self.po[1]+self.po[0]*xk, 'k--', linewidth=2.0)
        pylab.show()
        return self.po

    def save(self, n=500, filtercells=0, filterexp=0.0, annotation={}):
        """
        Computes RootedGraph.expr(), RootedGraph.delta(), RootedGraph.connectivity(),
        RootedGraph.connectivity_pvalue(), RootedGraph.centroid() and Benjamini-Holchberg adjusted q-values for all
        genes that are expressed in more than 'filtercells' cells and whose maximum expression
        value is above 'filterexp'. The optional argument 'annotation' allos to include a dictionary
        with lists of genes to be annotated in the table. The output is stored in a tab separated
        file called name.genes.txt.
        """
        pol = []
        with open(self.name + '.genes.tsv', 'w') as ggg:
            cul = 'Gene\tCells\tMean\tMin\tMax\tConnectivity\tp_value\tq-value (BH)\tCentroid\tDispersion\t'
            for m in sorted(annotation.keys()):
                cul += m + '\t'
            ggg.write(cul[:-1] + '\n')
            lp = sorted(self.dicgenes.keys())
            for gi in lp:
                if self.expr(gi) > filtercells and self.delta(gi)[2] > filterexp:
                    pol.append(self.connectivity_pvalue(gi, n=n))
            por = benjamini_hochberg(pol)
            mj = 0
            for gi in lp:
                po = self.expr(gi)
                m1, m2, m3 = self.delta(gi)
                p1, p2 = self.centroid(gi)
                if po > filtercells and m3 > filterexp:
                    cul = gi + '\t' + str(po) + '\t' + str(m1) + '\t' + str(m2) + '\t' + str(m3) + '\t' +\
                          str(self.connectivity(gi)) + '\t' + str(pol[mj]) + '\t' + str(por[mj]) +\
                          '\t' + str(p1) + '\t' + str(p2) + '\t'
                    for m in sorted(annotation.keys()):
                        if gi in annotation[m]:
                            cul += 'Y' + '\t'
                        else:
                            cul += 'N' + '\t'
                    ggg.write(cul[:-1] + '\n')
                    mj += 1
        centr = []
        disp = []
        centr2 = []
        disp2 = []
        f = open(self.name + '.genes.tsv', 'r')
        for n, line in enumerate(f):
            if n > 0:
                sp = line[:-1].split('\t')
                if float(sp[7]) <= 0.05:
                    centr.append(float(sp[1]))
                    disp.append(float(sp[5]))
                else:
                    centr2.append(float(sp[1]))
                    disp2.append(float(sp[5]))
        f.close()
        pylab.scatter(centr2, disp2, alpha=0.2, s=9, c='b')
        pylab.scatter(centr, disp, alpha=0.3, s=9, c='r')
        pylab.xlabel('cells')
        pylab.ylabel('connectivity')
        pylab.yscale('log')
        pylab.ylim(0.01, 1)
        pylab.xlim(0, max(centr+centr2))
        fig = pylab.figure()
        ax = fig.add_subplot(111, projection='3d')
        f = open(self.name + '.genes.tsv', 'r')
        for n, line in enumerate(f):
            if n > 0:
                sp = line[:-1].split('\t')
                if float(sp[7]) <= 0.05:
                    ax.scatter(float(sp[8]), float(sp[9]), float(sp[1]), c='k', alpha=0.2, s=10)
        ax.set_xlabel('Centroid')
        ax.set_ylabel('Dispersion')
        ax.set_zlabel('Cells')
        pylab.show()

    def cellular_subpopulations(self, min_dispersion, threshold=0.05, min_cells=5, max_K=8, method='centroid',
                                 clus_thres=0.65):
        """
        Identifies potential transient cellular subpopulations. The parameter 'min_dispersion'
        sets an upper bound of the dispersion of the genes that are considered in the analysis. The parameter
        'threshold' sets an upper bound of the q-value of the genes that are considered in the analysis.
        The parameter 'min_cells' sets the minimum number of cells on which each of the genes considered in the
        analysis is expressed. The parameter 'method' specifies the method used to determine transient populations.
        When 'method' is set to 'centroid', cellular subpopulations are determined by clustering the centroids of
        low dispersion genes using the results stored in the file 'name.genes.txt', produced by
        RootedGraph.save() The parameter 'max_K' sets the maximum number of clusters to be considered in the
        analysis. Two plots are produced. The first plot shows the dependence of the Davies-Bouldin index with
        respect to the number of clusters. The second plot displays the dispersion and centroid of genes that
        satisfy the threshold in the number of cells and the q-value, and shows the optimal clustering of low
        dispersion genes. When the parameter 'method' is set to 'js', cellular subpopulations are determined by
        clustering the Jensen-Shannon distance matrix of the genes that pass all the constraints. The number of
        clusters is controlled in this case by the parameter 'clus_thres'. In both cases a list with the genes
        associated to each cluster is returned. It requires the presence of the file 'name.genes.tsv', produced by the
        method RotedGraph.save().
        """
        con = []
        dis = []
        contot = []
        distot = []
        nam = []
        f = open(self.name + '.genes.tsv', 'r')
        for n, line in enumerate(f):
            if n > 0:
                sp = line[:-1].split('\t')
                if float(sp[9]) < min_dispersion and float(sp[7]) < threshold and float(sp[1]) > min_cells:
                    con.append(float(sp[8]))
                    dis.append(float(sp[9]))
                    nam.append(sp[0])
                elif float(sp[7]) < threshold and float(sp[1]) > min_cells:
                    contot.append(float(sp[8]))
                    distot.append(float(sp[9]))
        f.close()
        if method == 'centroid':
            y = []
            z = []
            for m in range(2, max_K):
                y_pred = sklearn.cluster.KMeans(n_clusters=m, random_state=170).fit_predict(numpy.array(
                    zip(con, [0.0]*len(con))))
                clus = [[] for _ in range(m)]
                for t, q in zip(y_pred, con):
                    clus[t].append(q)
                rij = []
                numpy.array(clus)
                for n1 in range(m):
                    rij.append([])
                    for n2 in range(m):
                        if n2 != n1:
                            rij[-1].append(
                                (numpy.var(clus[n1])-numpy.var(clus[n2]))/(abs(
                                    numpy.mean(clus[n1])-numpy.mean(clus[n2]))))
                        else:
                            rij[-1].append(0.0)
                ri = []
                for n in range(m):
                    ri.append(max(rij[n]))
                ssb = 0.0
                ssw = 0.0
                for n in range(m):
                    ssb += len(clus[n])*(numpy.mean(clus[n])-numpy.mean(con))**2
                    for q in clus[n]:
                        ssw += (q-numpy.mean(clus[n]))**2
                y.append(numpy.mean(ri))
                z.append(ssb*(len(con)-m)/(ssw*(m-1)))
            pylab.figure()
            pylab.plot(range(2, max_K), y, 'r-', linewidth=2.0)
            r = numpy.infty
            rn = -1
            for n, m in enumerate(y):
                if m < r:
                    rn = n
                    r = m
            y_pred = sklearn.cluster.KMeans(n_clusters=rn+2, random_state=170).fit_predict(
                numpy.array(zip(con, [0.0]*len(con))))
            pylab.figure()
            pylab.scatter(contot, distot, c='k', alpha=0.1)
            pylab.scatter(con, dis, c=y_pred, alpha=0.6)
            pylab.xlabel('Centroid')
            pylab.ylabel('Dispersion')
            pylab.show()
            g = [[] for _ in range(rn + 2)]
            for m, n in zip(list(y_pred), nam):
                g[m].append(n)
            return g
        elif method == 'js':
            return UnrootedGraph.cellular_subpopulations(self, min_dispersion, threshold=threshold, min_cells=min_cells,
                                                         clus_thres=clus_thres)
