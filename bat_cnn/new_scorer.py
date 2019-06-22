import numpy as np
import SimpleITK as sitk 
from matplotlib import pyplot as plt
import progressbar
import glob
import os
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
import statistics

truth_PREFIX = 'truth.nii.gz'
preds_PREFIX = 'prediction.nii.gz'
MASK_PREFIX = 'data_mask.nii.gz'
label_dict = {'1': 'boneLesion', '2':'lymphNodeLesion', '3': 'localProstateLesion'}
# label_dict = {'1': 'boneLesion', '2':'lymphNodeLesion'}
epslone = 1e-10

acc_area=25
rootdir = '/media/data/yuzhao/project/PSMA/Cross_Validation/resultData_4'
evaluation_list = glob.glob(rootdir+'/prediction*'+'/*_case_*')
evaluation_list.sort()

def get_itk_array(filename):
    image = sitk.ReadImage(filename)
    imageArray = sitk.GetArrayFromImage(image)
    return imageArray

def load_lesion_labels():
#    data = []
    for pn in evaluation_list:
        fnl = os.path.join(pn, truth_PREFIX)
        fnm = os.path.join(pn, MASK_PREFIX)
        l = get_itk_array(fnl)
        m = get_itk_array(fnm)
        m = np.asarray((m>0),dtype='int')
        yield(np.asarray((l * m), dtype='int'))

def load_predictions(preds_PREFIX):
#    data = []
    for pn in evaluation_list:
        fnl = os.path.join(pn, preds_PREFIX)
        fnm = os.path.join(pn, MASK_PREFIX)
        l = get_itk_array(fnl)
        m = get_itk_array(fnm)
        m = np.asarray((m>0),dtype='int')
        yield(np.asarray(l * m, dtype='int'))

def get_regions(gtslice, labels):
    dummy = np.zeros(gtslice.shape, dtype='int')
    dslices = {}
    cnts = {}
    for label in labels:
        dslices[str(label)] = np.copy(dummy)
        cnts[str(label)] = 0

    inds = np.where(np.isin(gtslice,labels))

    for x,y in zip(inds[0], inds[1]):
        label = gtslice[x,y]
        if dslices[str(label)][x,y] == 0:
            thisRegion = np.zeros(gtslice.shape, dtype='int')
            temp = np.zeros(gtslice.shape, dtype='int')
            thisRegion[x,y] = 1
            new_ind = np.where(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int') == 1)
            iterate = np.sum(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int')) > 0
            temp = np.copy(thisRegion)

            while iterate:
                for xi, yi in zip(new_ind[0], new_ind[1]):
                    patch = gtslice[xi-1:xi+2,yi-1:yi+2]
                    patch = np.asarray(patch == label, dtype='int')
                    thisRegion[xi-1:xi+2,yi-1:yi+2] = patch

                iterate = np.sum(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int')) > 0
                if iterate:
                    new_ind = np.where(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int') == 1)

                temp = np.copy(thisRegion)
            cnts[str(label)] += 1
            dslices[str(label)] = dslices[str(label)] + thisRegion * cnts[str(label)]

    return dslices, cnts

def get_regions_3d(gtslice, labels):
    dummy = np.zeros(gtslice.shape, dtype='int')
    dslices = {}
    cnts = {}
    for label in labels:
        dslices[str(label)] = np.copy(dummy)
        cnts[str(label)] = 0

    inds = np.where(np.isin(gtslice,labels))

    for x,y,z in zip(inds[0], inds[1], inds[2]):
        label = gtslice[x,y,z]
        if dslices[str(label)][x,y,z] == 0:
            thisRegion = np.zeros(gtslice.shape, dtype='int')
            temp = np.zeros(gtslice.shape, dtype='int')
            thisRegion[x,y,z] = 1
            new_ind = np.where(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int') == 1)
            iterate = np.sum(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int')) > 0
            temp = np.copy(thisRegion)

            while iterate:
                for xi, yi, zi in zip(new_ind[0], new_ind[1], new_ind[2]):
                    patch = gtslice[xi-1:xi+2,yi-1:yi+2,zi-1:zi+2]
                    patch = np.asarray(patch == label, dtype='int')
                    thisRegion[xi-1:xi+2,yi-1:yi+2,zi-1:zi+2] = patch

                iterate = np.sum(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int')) > 0
                if iterate:
                    new_ind = np.where(np.asarray(temp == 0, dtype='int') * np.asarray(thisRegion == 1, dtype='int') == 1)

                temp = np.copy(thisRegion)
            cnts[str(label)] += 1
            dslices[str(label)] = dslices[str(label)] + thisRegion * cnts[str(label)]
    return dslices, cnts

def get_counts(y_true, y_pred, labels, percent = 0.5, acc_area=acc_area):
    scores = {}
    tot_slices = y_true.shape[0]
    bar = progressbar.ProgressBar(max_value=tot_slices)
    slice_cnt = 0
    for grt, pred in zip(y_true, y_pred):
        rgrts,rgcnts = get_regions(grt,labels)
        rpreds,rpcnts = get_regions(pred,labels)

        for label in labels:
            rgrt = rgrts[str(label)]
            rgcnt = rgcnts[str(label)]
            rpred = rpreds[str(label)]
            rpcnt = rpcnts[str(label)]

            TP = 0
            FP = 0
            FN = 0
            TN = 0

            for i in np.arange(1,rgcnt+1, 1):
                overlap = np.sum(np.asarray(rgrt == i, dtype='int') * np.asarray(pred == label)) * 1.0 / np.sum(np.asarray(rgrt == i, dtype='int'))
                area = np.sum(np.asarray(rgrt == i, dtype='int'))
                if overlap >= percent and area > acc_area:
                    TP += 1
                elif area > acc_area:
                    FN += 1

            for i in np.arange(1,rpcnt+1, 1):
                overlap = np.sum(np.asarray(rpred == i, dtype='int') * np.asarray(grt == label)) * 1.0 / np.sum(np.asarray(rpred == i, dtype='int'))
                area = np.sum(np.asarray(rpred == i, dtype='int'))
                if overlap < percent and area > acc_area:
                    FP += 1

            score = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

            if str(label) in scores:
                scores[str(label)]['TP'] += TP
                scores[str(label)]['FP'] += FP
                scores[str(label)]['TN'] += TN
                scores[str(label)]['FN'] += FN
            else:
                scores[str(label)] = score

        slice_cnt += 1
        bar.update(slice_cnt)
    return scores

def get_counts_3d(y_true, y_pred, labels, percent = 0.5, acc_area=acc_area):
    scores = {}

    rgrts,rgcnts = get_regions_3d(y_true,labels)
    rpreds,rpcnts = get_regions_3d(y_pred,labels)


    for label in labels:
        rgrt = rgrts[str(label)]
        rgcnt = rgcnts[str(label)]
        rpred = rpreds[str(label)]
        rpcnt = rpcnts[str(label)]

        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for i in np.arange(1,rgcnt+1, 1):
            overlap = np.sum(np.asarray(rgrt == i, dtype='int') * np.asarray(y_pred == label)) * 1.0 / np.sum(np.asarray(rgrt == i, dtype='int'))
            area = np.sum(np.asarray(rgrt == i, dtype='int'))
            if overlap >= percent and area > acc_area:
                TP += 1
            elif area > acc_area:
                FN += 1

        for i in np.arange(1,rpcnt+1, 1):
            overlap = np.sum(np.asarray(rpred == i, dtype='int') * np.asarray(y_true == label)) * 1.0 / np.sum(np.asarray(rpred == i, dtype='int'))
            area = np.sum(np.asarray(rpred == i, dtype='int'))
            if overlap < percent and area > acc_area:
                FP += 1

        score = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

        if str(label) in scores:
            scores[str(label)]['TP'] += TP
            scores[str(label)]['FP'] += FP
            scores[str(label)]['TN'] += TN
            scores[str(label)]['FN'] += FN
        else:
            scores[str(label)] = score
    return scores

def get_count_from_lists(y_true_list, y_pred_list, labels, labels_3d=None, percent=0.5):
    acc_scores = []
    cnt = 0
    print ('Calculating Scores .....')
    for y_true, y_pred in zip(y_true_list,y_pred_list):
        cnt += 1
        print ('Patient', cnt)
        scores = get_counts(y_true, y_pred, labels, percent)
        if not labels_3d is None:
            scores_3d = get_counts_3d(y_true, y_pred, labels_3d, percent)
            for lab in labels_3d:
                scores[str(lab)] = scores_3d[str(lab)]

        acc_scores.append(scores)
    return acc_scores

def calculate_scores(count_list, label_dict):
    global_scores = {}
    patient_scores = {}
    patient_combined_scores = {'precision': [],'recall': [],'dice': [],'accuracy': []}
    for pat, scores in enumerate(count_list):
        print ('patient :', pat + 1)
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for key in scores:
            if not key in global_scores:
                global_scores[key] = {'TP': 0,'FP': 0,'FN': 0,'TN': 0}
                patient_scores[key] = {'precision': [],'recall': [],'dice': [],'accuracy': []}

            sc = scores[key]

            if sc['TP'] + sc['FN'] > 0:

                accuracy = (sc['TP'] + sc['TN'])*1.0 / (sc['TP'] + sc['TN'] + sc['FP'] + sc['FN'] + epslone)
                precision = 0 if sc['TP'] == 0 else sc['TP']*1.0 / (sc['TP'] + sc['FP'] + epslone)
                recall = sc['TP']*1.0 / (sc['TP'] + sc['FN']+ epslone)
                dice = 0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall + epslone)

                print (label_dict[key],':')
                print ( 'precision', precision)
                print ( 'recall', recall)
                print ( 'dice', dice)
                print ( 'accuracy', accuracy)
                print ( 'total', sc['TP'] + sc['FN'])

                patient_scores[key]['precision'].append(precision)
                patient_scores[key]['recall'].append(recall)
                patient_scores[key]['dice'].append(dice)
                patient_scores[key]['accuracy'].append(accuracy)

                TP += sc['TP']
                TN += sc['TN']
                FP += sc['FP']
                FN += sc['FN']

                global_scores[key]['TP'] += sc['TP']
                global_scores[key]['TN'] += sc['TN']
                global_scores[key]['FP'] += sc['FP']
                global_scores[key]['FN'] += sc['FN']

        accuracy = 0 if TP == 0 else (TP + TN) * 1.0 / (TP + TN + FP + FN + epslone)
        precision = 0 if TP == 0 else TP * 1.0 / (TP + FP + epslone)
        recall = 0 if TP == 0 else TP * 1.0 / (TP + FN + epslone)
        dice = 0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall + epslone)

        print ('patient summary')
        print ( 'precision', precision)
        print ( 'recall', recall)
        print ( 'dice', dice)
        # print ( 'accuracy', accuracy)

        patient_combined_scores['precision'].append(precision)
        patient_combined_scores['recall'].append(recall)
        patient_combined_scores['dice'].append(dice)
        patient_combined_scores['accuracy'].append(accuracy)

    print ('Patient Scores')
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for lab in global_scores:
        sc = global_scores[lab]
        # accuracy = sc['TP']*1.0 / (sc['TP'] + sc['FN'] + epslone)
        accuracy = (sc['TP'] + sc['TN'])*1.0 / (sc['TP'] + sc['TN'] + sc['FP'] + sc['FN'] + epslone)
        precision = 0 if sc['TP'] == 0 else sc['TP']*1.0 / (sc['TP'] + sc['FP'] + epslone)
        recall = sc['TP']*1.0 / (sc['TP'] + sc['FN'] + epslone)
        dice = 0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall + epslone)

        # print (label_dict[lab],' accuracy:',accuracy, ' precision:',precision, ' recall:',recall, ' dice:',dice, '  total:', sc['TP'] + sc['FN'])
        print (label_dict[lab], ' precision:',precision, ' recall:',recall, ' dice:',dice, '  total:', sc['TP'] + sc['FN'])
        TP += sc['TP']
        TN += sc['TN']
        FP += sc['FP']
        FN += sc['FN']

    accuracy = 0 if TP == 0 else (TP + TN) * 1.0 / (TP + TN + FP + FN + epslone)
    precision = 0 if TP == 0 else TP * 1.0 / (TP + FP + epslone)
    recall = 0 if TP == 0 else TP * 1.0 / (TP + FN + epslone)
    dice = 0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall + epslone)
    # print ('Combined',' accuracy:',accuracy, ' precision:',precision, ' recall:',recall, ' dice:',dice, '  total:', TP + FN)
    return patient_scores, patient_combined_scores, global_scores

def box_plot(patient_scores, patient_combined_scores,label_dict,labels):

   # function for setting the colors of the box plots pairs
   def setBoxColors(bp):
       setp(bp['boxes'][0], color='blue')
       setp(bp['caps'][0], color='blue')
       setp(bp['caps'][1], color='blue')
       setp(bp['whiskers'][0], color='blue')
       setp(bp['whiskers'][1], color='blue')
       setp(bp['medians'][0], color='blue')

       setp(bp['boxes'][1], color='red')
       setp(bp['caps'][2], color='red')
       setp(bp['caps'][3], color='red')
       setp(bp['whiskers'][2], color='red')
       setp(bp['whiskers'][3], color='red')
       setp(bp['medians'][1], color='red')

       setp(bp['boxes'][2], color='green')
       setp(bp['caps'][4], color='green')
       setp(bp['caps'][5], color='green')
       setp(bp['whiskers'][4], color='green')
       setp(bp['whiskers'][5], color='green')
       setp(bp['medians'][2], color='green')

   # data prepare:
   group_data = {}
   for key in label_dict:
       group_data[key] = [patient_scores[key]['precision'],patient_scores[key]['recall'],patient_scores[key]['dice']]
    
   group_data['combined'] = [patient_combined_scores['precision'], patient_combined_scores['recall'], patient_combined_scores['dice']]      

   fig = figure()
   ax = axes()
#   hold(True)

   # first boxplot pair
   bp = boxplot(group_data[str(labels[0])], positions = [1,2,3], widths = 0.3, showfliers = False)
   setBoxColors(bp)

   # second boxplot pair
   bp = boxplot(group_data[str(labels[1])], positions = [5,6,7], widths = 0.3, showfliers = False)
   setBoxColors(bp)

   # thrid boxplot pair
   bp = boxplot(group_data['combined'], positions = [9,10,11], widths = 0.3, showfliers = False)
   setBoxColors(bp)

   # set axes limits and labels
   xlim(0,12)
   ylim(0.0,1.1)
   ax.set_xticklabels(['Bone Lesion', 'Lymph Node Lesion', 'Combined'])
   ax.set_xticks([2, 6, 10])

   # draw temporary red and blue lines and use them to create a legend
   hB, = plot([1,1],'b-')
   hR, = plot([1,1],'r-')
   hY, = plot([1,1],'g-')
   legend((hB, hR, hY),('Precision', 'Recall', 'F1 Score'))
   hB.set_visible(False)
   hR.set_visible(False)
   hY.set_visible(False)

   savefig(os.path.join(rootdir,'boxcompare.png'))
   # show()


#   print(label_dict[str(1)])
#   print('Precision')
#   print('mean:'+str(statistics.mean(group_data[str(labels[0])][0]))+',SDV:' + str(statistics.stdev(group_data[str(labels[0])][0])))
#   print('Recall')
#   print('mean:'+str(statistics.mean(group_data[str(labels[0])][1]))+',SDV:' + str(statistics.stdev(group_data[str(labels[0])][1])))
#   print('F1 Score')
#   print('mean:'+str(statistics.mean(group_data[str(labels[0])][2]))+',SDV:' + str(statistics.stdev(group_data[str(labels[0])][2])))
#
#   print(label_dict[str(2)])
#   print('Precision')
#   print('mean:'+str(statistics.mean(group_data[str(labels[1])][0]))+',SDV:' + str(statistics.stdev(group_data[str(labels[1])][0])))
#   print('Recall')
#   print('mean:'+str(statistics.mean(group_data[str(labels[1])][1]))+',SDV:' + str(statistics.stdev(group_data[str(labels[1])][1])))
#   print('F1 Score')
#   print('mean:'+str(statistics.mean(group_data[str(labels[1])][2]))+',SDV:' + str(statistics.stdev(group_data[str(labels[1])][1])))


if __name__ == '__main__':
    labels = load_lesion_labels()
    print ('SCORES CF VNET')    
    preds = load_predictions(preds_PREFIX)
    scores = get_count_from_lists(y_true_list=labels, y_pred_list=preds, labels=[1,2,3], labels_3d=None, percent = 0.1)
    patient_scores, patient_combined_scores, global_scores = calculate_scores(scores, label_dict)
    box_plot(patient_scores, patient_combined_scores,label_dict, labels=[1,2])
