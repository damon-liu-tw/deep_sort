# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    @property
    def dict(self):
        tlbr = self.to_tlbr()
        l,t,r,b = [float(f'{n:.6f}') for n in tlbr]
        return dict(xmin=l, xmax=r, ymin=t, ymax=b, label='r', conf=1)
    
    @classmethod
    def from_dict(cls, obj_dict):
        tlwh = (obj_dict['xmin'],
                obj_dict['ymin'],
                obj_dict['xmax'] - obj_dict['xmin'],
                obj_dict['ymax'] - obj_dict['ymin'])
        
        return cls(tlwh, obj_dict['conf'], feature=None)
    
    @classmethod
    def from_tlbr(cls, tlbr):
        h = tlbr[2] - tlbr[0]
        w = tlbr[3] - tlbr[1]
        return cls([tlbr[1], tlbr[0], w, h], 1, None)
