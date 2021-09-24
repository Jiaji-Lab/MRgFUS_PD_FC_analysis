import ast
import csv
import os
import xml.etree.ElementTree as ET

import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd
from nilearn.image import smooth_img
from scipy.io import loadmat

from . import mask
from . import utils

class Modal(object):
    """
    Parent class for all modal.
    Attributes:
        directory: directory of this modal.
        mark: mark shows in each file in this modal, e.g. "t1" for "t1.nii" and "cat_t1.xml".
    Functions:
        build_path: generate full path to file.
        remove_file: remove centain file in this modal.
        load_nii: load nii.
        load_csv: load csv.
    """
    def __init__(self, directory, mark):
        self.directory = directory
        self.mark = mark
        self.metrics = None

    def build_path(self, filename, use_mark=True):
        if use_mark:
            path = os.path.join(self.directory, filename.format(self.mark))
        else:
            path = os.path.join(self.directory, filename)
        return path

    def create_roi_volume_csv(self, mask,
                              image_path, out_csv_path,
                              feature_name='Volume'):
        """
        Use mask to calculate ROI volume then store to csv
        Args:
            mask: Mask instance, ROI template use to calculate volume
            prefix: string, path to nii file with placeholder for #mark#
            out_csv_path: string, path to store ROI GMV csv with placeholder for #mark#
            feature_name: string, csv colume name for feature
        """
        image = self.load_image(image_path)

        volumes = mask.get_all_masked_volume(image)

        with open(out_csv_path, 'w', newline='') as file:
            fieldnames = ['ID', feature_name]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for k, v in volumes.items():
                writer.writerow({'ID': k, feature_name: v})
    
    def create_roi_mean_csv(self, mask,
                            image_path, out_csv_path,
                            feature_name='Mean'):
        """
        Use mask to calculate ROI mean then store to csv
        Args:
            mask: Mask instance, ROI template use to calculate volume
            prefix: string, path to nii file with placeholder for #mark#
            out_csv_path: string, path to store ROI GMV csv with placeholder for #mark#
            feature_name: string, csv colume name for feature
        """
        image = self.load_image(image_path)

        volumes = mask.get_all_masked_mean(image)

        with open(out_csv_path, 'w', newline='') as file:
            fieldnames = ['ID', feature_name]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for k, v in volumes.items():
                writer.writerow({'ID': k, feature_name: v})

    def create_smoothed_nii(self, image_path, out_path, fwhm=4):
        """
        create smoothed nii then store it.
        Args:
            prefix: string, path to nii file with placeholder for #mark#
            fwhm: smooth kernel size
        Returns:
            smoothed nibabel Nifti1Image instance
        """
        nii = self.load_image(image_path)
        smoothed_nii = smooth_img(nii, fwhm)
        nib.save(smoothed_nii, out_path)
        return smoothed_nii

    def load_csv(self, csv_path, **kwargs):
        """
        load csv
        Args:
            csv_path: string, full path to *.csv file
            **kwargs: pass to pd.read_csv()
        Returns:
            dataframe
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        return pd.read_csv(csv_path, **kwargs)

    def load_image(self, image_path):
        """
        load image
        Args:
            image_path: string, full path to *.nii/*.gii file
        Returns:
            nibabel Nifti1Image/GiftiImage instance, 
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        return nib.load(image_path)

    def rename(self, src_name, to_name, use_mark=True):
        src_path = self.build_path(src_name, use_mark)
        to_path = self.build_path(to_name, use_mark)
        os.rename(src_path, to_path)

    def remove_file(self, path):
        os.remove(path)

    def load_graph_metrics(self, mat_name='brant.mat',
                           threshold_type='intensity', threshold_value=0.3):
        mat_path = self.build_path(mat_name, use_mark=False)
        mat = loadmat(mat_path)
        if threshold_type == 'intensity':
            threshold_values = mat['thres_corr_use'].flatten().tolist()
            all_metrics = mat['calc_rsts_corr']
        elif threshold_type == 'sparsity':
            threshold_values = mat['thres_spar_use'].flatten().tolist()
            all_metrics = mat['calc_rsts_spar']

        i = threshold_values.index(threshold_value)
        self.metrics = all_metrics[i][0]
        return self.metrics

    def get_global_metric(self, metric_name,
                          mat_name='brant.mat',
                          threshold_type='intensity', threshold_value=0.3):
        """
        get global metric from loaded metrics
        Args:
            metric_name: string, one of the following:
                            [assortative_mixing, neighbor_degree,
                            betweenness_centrality, degree,
                            fault_tolerance, shortest_path_length,
                            global_efficiency, clustering_coefficient,
                            local_efficiency, resilience,
                            vulnerability, transitivity]
            mat_name: string, pass to #load_graph_metrics#
            threshold_type: string, pass to #load_graph_metrics#
            threshold_value: float, pass to #load_graph_metrics#
        Returns:
            metric: int, global metric. when 'resilience', return ndarray.
        """
        metrics = self.load_graph_metrics(mat_name,
                                            threshold_type,
                                            threshold_value)
        if metric_name == 'resilience':
            # special case, (n, 2) array
            metric = metrics[metric_name][0][0][0][0][0]
        else:
            metric = metrics[metric_name][0][0][0][0][0][0][0]
        return metric

    def get_nodal_metric(self, metric_name,
                         mat_name='brant.mat',
                         threshold_type='intensity', threshold_value=0.3):
        """
        get nodal metric from loaded metrics
        Args:
            metric_name: string, one of the following:
                            [neighbor_degree,
                            betweenness_centrality, degree,
                            fault_tolerance, shortest_path_length,
                            global_efficiency, clustering_coefficient,
                            local_efficiency, vulnerability]
            mat_name: string, pass to #load_graph_metrics#
            threshold_type: string, pass to #load_graph_metrics#
            threshold_value: float, pass to #load_graph_metrics#
        Returns:
            metric: ndarray, (n,) for n ROIs
        """
        if self.metrics is None:
            metrics = self.load_graph_metrics(mat_name,
                                              threshold_type,
                                              threshold_value)
        else:
            metrics = self.metrics
        metric = metrics[metric_name][0][0][0][0][1].flatten()
        return metric

class Asl(Modal):
    """
    T1 sMRI feature management
    Attributes:
        directory: directory of T1
        mark: mark shows in each file, e.g. "t1" for "t1.nii" and "cat_t1.xml"
    Functions:
        create_roi_ct: create a csv contains ROI cortical thickness from cat12 catROIs_{}.xml file
        create_roi_gmv: create a csv contains ROI grey matter volume using mask
        create_smoothed_nii: smooth nii then store it.
        get_image_quailty: get image quailty generated by cat12 segment pipeline from cat_{}.xml
        get_total_features: get total TIV, GMV, WMV, CSF generated by cat12 segment pipeline from cat_{}.xml
    """
    def __init__(self, directory, mark='asl',
                 lamda=0.9, pld=2.0, t1_blood=1.65,
                 alpha=0.85, tau=1.8):
        super().__init__(directory, mark)
        self.lamda = lamda
        self.pld = pld
        self.t1_blood = t1_blood
        self.alpha = alpha
        self.tau = tau
        self.ratio = 100 * self.lamda * np.exp(self.pld/self.t1_blood) / (2 * self.alpha * self.t1_blood * (1 - np.exp(-self.tau/self.t1_blood)))

    def create_roi_mean_csv(self, mask,
                            image_name, out_csv_name,
                            feature_name='Mean'):
        image_path = self.build_path(image_name, use_mark=False)
        out_csv_path = self.build_path(out_csv_name, use_mark=False)
        super().create_roi_mean_csv(mask, image_path, out_csv_path, feature_name)

    
    def generate_cbf_from_asl(self, asl_name='asl_mni.nii',
                         pd_name='pd_mni.nii',
                         cbf_out_name='cbf_mni.nii'):
        asl_path = self.build_path(asl_name, use_mark=False)
        pd_path = self.build_path(pd_name, use_mark=False)
        cbf_out_path = self.build_path(cbf_out_name, use_mark=False)
        asl_nii = nib.load(asl_path)
        pd_nii = nib.load(pd_path)
        asl_array = np.array(asl_nii.dataobj)
        pd_array = np.array(pd_nii.dataobj)

        data = self.ratio * asl_array / pd_array
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0
        utils.gen_nii(data, asl_nii, cbf_out_path)

    def threshold_cbf(self, cbf_name='cbf_mni.nii',
                      cbf_thres_name='cbf_mni_thres.nii',
                      ratio=1.5):
        cbf_path = self.build_path(cbf_name, use_mark=False)
        cbf_thres_path = self.build_path(cbf_thres_name, use_mark=False)
        cbf_nii = nib.load(cbf_path)
        data = np.array(cbf_nii.dataobj)
        thres_value = ratio * np.median(data[data.nonzero()])
        thres_data = data * (data<=thres_value)
        utils.gen_nii(thres_data, cbf_nii, cbf_thres_path)

    def load_csv(self, csv_name, **kwargs):
        csv_path = self.build_path(csv_name, use_mark=False)
        return super().load_csv(csv_path)

class T1(Modal):
    """
    T1 sMRI feature management
    Attributes:
        directory: directory of T1
        mark: mark shows in each file, e.g. "t1" for "t1.nii" and "cat_t1.xml"
    Functions:
        create_roi_ct: create a csv contains ROI cortical thickness from cat12 catROIs_{}.xml file
        create_roi_gmv: create a csv contains ROI grey matter volume using mask
        create_smoothed_nii: smooth nii then store it.
        get_image_quailty: get image quailty generated by cat12 segment pipeline from cat_{}.xml
        get_total_features: get total TIV, GMV, WMV, CSF generated by cat12 segment pipeline from cat_{}.xml
    """
    def __init__(self, directory, mark='t1'):
        super().__init__(directory, mark)

    def create_smoothed_nii(self, image_name, out_filename, fwhm=4):
        image_path = self.build_path(image_name)
        out_path = self.build_path(out_filename)
        super().create_smoothed_nii(image_path, out_path)

    def create_roi_ct_from_cat(self, 
                               filename='label/catROIs_{}.xml',
                               out_csv_filename='label/roi_ct_{}.csv',
                               atlas_name='aparc_BN_Atlas',
                               cortical_id_path='../data/mask/brainnetome/cortical_id.csv'):
        """
        create a csv contains ROI cortical thickness from cat12 catROIs_{}.xml file
        Args:
            filename: string, path to catROIs_{}.xml file with placeholder for #mark#
            out_csv_filename: string, path to store ROI CT csv with placeholder for #mark#
            atlas_name: string, atlas name in catROIs file
            cortical_id_path: string, path to atlas name-id chart
        """
        report = ET.parse(self.build_path(filename))
        root = report.getroot()
        names = root.findall('./{}/names'.format(atlas_name))

        thickness = root.find('./{}/data/thickness'.format(atlas_name))
        thickness = thickness.text.replace(' ', ',')
        thickness = thickness.replace('NaN', '-1')
        thickness_list = ast.literal_eval(thickness)

        out_csv_path = self.build_path(out_csv_filename)
        id_df = pd.read_csv(cortical_id_path, index_col=0)
        with open(out_csv_path, 'w', newline='') as file:
            fieldnames = ['ID', 'CT']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for (item, thickness) in zip(names[0].findall('item'), thickness_list):
                name = item.text
                if name[0].lower() == name[-1].lower():
                    # special case for BN atlas
                    name = name[1:]
                    if '/' in name:
                        name = name.replace('/', '-')
                    writer.writerow({'ID': id_df.loc[name]['ID'], 'CT': thickness})

    def create_roi_volume_csv(self, mask,
                              image_name, out_csv_filename,
                              feature_name='Volume'):
        image_path = self.build_path(image_name)
        out_csv_path = self.build_path(out_csv_filename)
        super().create_roi_volume_csv(mask, image_path, out_csv_path, feature_name)

    def get_image_quailty(self, filename='report/cat_{}.xml'):
        """
        Get all image qualties measures rated by cat12 segment pipeline
        Args:
            filename: string, path to cat_{}.xml file with placeholder for #mark#
        Returns:
            qualties: dict, contains all image qualties measures
        """
        xml_path = self.build_path(filename)
        report = ET.parse(xml_path)
        root = report.getroot()

        resolution = 10.5 - float(root.findall('./qualityratings/res_RMS')[0].text)
        noise = 10.5 - float(root.findall('./qualityratings/NCR')[0].text)
        bias = 10.5 - float(root.findall('./qualityratings/ICR')[0].text)
        iqr = 10.5 - float(root.findall('./qualityratings/IQR')[0].text)

        qualties = {}
        qualties['Resolution'] = resolution
        qualties['Noise'] = noise
        qualties['Bias'] = bias
        qualties['IQR'] = iqr

        return qualties

    def get_total_features(self, filename='report/cat_{}.xml'):
        """
        Get all whole brain features(TIV, GMV, WMV, CSF) calculated by cat12 segment pipeline
        Args:
            filename: string, path to cat_{}.xml file with placeholder for #mark#
        Returns:
            features: dict, contains all whole brain features
        """
        xml_path = self.build_path(filename)
        report = ET.parse(xml_path)
        root = report.getroot()

        tiv = root.findall('./subjectmeasures/vol_TIV')[0].text
        cgw = root.findall('./subjectmeasures/vol_abs_CGW')
        tmp = [float(i) for i in cgw[0].text.replace('[', '').replace(']', '').split()]

        features = {}
        features['TIV'] = tiv
        features['CSF'] = tmp[0]
        features['GMV'] = tmp[1]
        features['WMV'] = tmp[2]

        return features

    def load_csv(self, csv_name, **kwargs):
        csv_path = self.build_path(csv_name)
        return super().load_csv(csv_path)

class Dti(Modal):
    def __init__(self, directory, mark='dti'):
        super().__init__(directory, mark)
        self.network_fa = None
        self.network_md = None
        self.network_num = None

    def create_roi_volume_csv(self, image_name, out_csv_filename,
                              mask_name='BN_Atlas_246_1mm.nii.gz',
                              feature_name='Volume'):
        mask = self.load_individual_mask(mask_name)
        image_path = self.build_path(image_name, use_mark=False)
        out_csv_path = self.build_path(out_csv_filename, use_mark=False)
        super().create_roi_volume_csv(mask, image_path, out_csv_path, feature_name)

    def load_network(self, filename='network_num.txt'):
        txt_path = self.build_path(filename, use_mark=False)
        structural_network = np.loadtxt(txt_path)
        return structural_network

    def load_individual_mask(self, mask_name):
        mask_path = self.build_path(mask_name, use_mark=False)
        return mask.NiiMask(mask_path)

    def get_network_fa(self):
        if self.network_fa is None:
            self.network_fa = self.load_network(filename='network_fa.txt')
        return self.network_fa

    def get_network_md(self):
        if self.network_md is None:
            self.network_md = self.load_network(filename='network_md.txt')
        return self.network_md

    def get_network_num(self):
        if self.network_num is None:
            self.network_num = self.load_network(filename='network_num.txt')
        return self.network_num

    def get_ave_control(self, mat_name='controllablity.mat'):
        mat_path = self.build_path(mat_name, use_mark=False)
        mat = loadmat(mat_path)
        ave_control = mat['controllablity']['ave'][0][0].flatten()
        return ave_control
    
    def get_modal_control(self, mat_name='controllablity.mat'):
        mat_path = self.build_path(mat_name, use_mark=False)
        mat = loadmat(mat_path)
        ave_control = mat['controllablity']['modal'][0][0].flatten()
        return ave_control

    def load_csv(self, csv_name, **kwargs):
        csv_path = self.build_path(csv_name, use_mark=False)
        return super().load_csv(csv_path, kwargs)