import os
import sys
from newLayers import *
cwd = os.getcwd()
sys.path.append(cwd+'/../')

class Mask():
    def __init__(self, model, threshold):
        self.target = []
        self.mask = []
        for m in model.modules():
            if isinstance(m, Winograd2d):
                self.target.append(m)
                self.mask.append(m.weight.data.clone().abs().lt(threshold))
        return

    def print_info(self):
        print '-------------------------------------'
        for i in range(len(self.mask)):
            mask = self.mask[i]
            print '[{}]: {} / {} ( {:.2f}% )'.format(i, mask.sum(), mask.nelement(), 100. * float(mask.sum()) / mask.nelement())
        print '-------------------------------------'
        return

    def apply(self):
        for i in range(len(self.mask)):
            self.target[i].weight.data[self.mask[i]] = 0.0
        return
