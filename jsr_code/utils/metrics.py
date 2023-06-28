import numpy as np
from scipy.interpolate import interp1d

class Evaluator(object):
    def reset(self):
        """ Reset internal variables between epochs (or validation runs) """
        raise NotImplementedError
    def add_batch(self, gt_image, pre_image, **kwargs):
        """ Add data from batch for matric accumulation """
        raise NotImplementedError
    def compute_stats(self, **kwargs):
        """ Compute/print metric and return main matric valie """
        raise NotImplementedError


class SegmEvaluator(Evaluator):
    def __init__(self, cfg, num_class, **kwargs):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image_t, output):
        gt_image = gt_image_t.cpu().numpy()
        pred = output.data.cpu().numpy()
        pre_image = np.argmax(pred, axis=1)
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def compute_stats(self, **kwargs):
        Acc = self.Pixel_Accuracy()
        Acc_class = self.Pixel_Accuracy_Class()
        mIoU = self.Mean_Intersection_over_Union()
        FWIoU = self.Frequency_Weighted_Intersection_over_Union()

        writer = kwargs.get("writer", None)
        if writer is not None: 
            epoch = kwargs.get("epoch", 0)
            writer.add_scalar('val/mIoU', mIoU, epoch)
            writer.add_scalar('val/Acc', Acc, epoch)
            writer.add_scalar('val/Acc_class', Acc_class, epoch)
            writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        return mIoU


class AnomalyEvaluator(Evaluator):
    def __init__(self, cfg, **kwargs):
        self.ignore_label = cfg.LOSS.IGNORE_LABEL
        self.quantization = 1000
        self.anomaly_range = [0, 1]
        self.cmat = np.zeros(shape=[self.quantization, 2, 2])  
    
    def create_roi(self, gt_label):
        roi = (gt_label != self.ignore_label).astype(np.bool)
        # TODO limit roi to regions around road, so the metric is focused on anomalies on the road
        return roi

    def compute_cmat(self, gt_label, prob):
        roi = self.create_roi(gt_label)
        prob = prob[roi]
        area = prob.__len__()
        gt_label = gt_label[roi]

        gt_mask_road = (gt_label == 1)
        gt_mask_obj = ~gt_mask_road

        gt_area_true = np.count_nonzero(gt_mask_obj)
        gt_area_false = area - gt_area_true

        prob_at_true = prob[gt_mask_obj]
        prob_at_false = prob[~gt_mask_obj]

        tp, _ = np.histogram(prob_at_true, self.quantization, range=self.anomaly_range)
        tp = np.cumsum(tp[::-1])

        fn = gt_area_true - tp

        fp, _ = np.histogram(prob_at_false, self.quantization, range=self.anomaly_range)
        fp = np.cumsum(fp[::-1])

        tn = gt_area_false - fp

        cmat = np.array([
            [tp, fp],
            [fn, tn],
            ]).transpose(2, 0, 1)
        
        if area > 0:
            cmat = cmat.astype(np.float64) / area
        else:
            cmat[:] = 0

        if np.any((cmat>1) | (cmat<0)):
            assert False, "Error in computing tp,fp,fn,tn. Some values larger than 1 or less than 0 {}".format(cmat)

        return cmat, area > 0

    def add_batch(self, gt_image_t, output):
        gt_image = gt_image_t.cpu().numpy()
        pre_image = output["anomaly_score"].cpu().numpy()[:, 0, ...]
        assert gt_image.shape == pre_image.shape
        for b in range(0, pre_image.shape[0]):
            cmat_b, valid_frame = self.compute_cmat(gt_image[b, ...], pre_image[b, ...])
            if valid_frame:
                self.cmat += cmat_b

    def reset(self):
        self.cmat[:] = 0

    def compute_stats(self, **kwargs):
        tp = self.cmat[:, 0, 0]
        fp = self.cmat[:, 0, 1]
        fn = self.cmat[:, 1, 0]
        tn = self.cmat[:, 1, 1]

        tp_rates = tp / (tp+fn) # = recall
        fp_rates = fp / (fp+tn)

        fp[(tp+fp) == 0] = 1e-9
        precision = tp / (tp+fp) 

        area_under_TPRFPR = np.trapz(tp_rates, fp_rates)
        AP = np.trapz(precision, tp_rates)

        f = interp1d(tp_rates, fp_rates, kind="linear")
        FPRat95 = f(0.95)

        writer = kwargs.get("writer", None)
        if writer is not None: 
            epoch = kwargs.get("epoch", 0)
            writer.add_scalar('val/AP', AP, epoch)
            writer.add_scalar('val/FPR@95', FPRat95, epoch)
        print("AP:{}, FPR@95:{}".format(AP, FPRat95))
        flag = kwargs.get("return_all", False)
        if flag:
            return [AP, FPRat95]
        else:
            return AP 




