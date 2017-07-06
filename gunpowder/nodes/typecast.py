import logging

import numpy as np

from .batch_filter import BatchFilter

logger = logging.getLogger(__name__)


class Typecast(BatchFilter):
    def __init__(self, volume_dtypes=None, safe=False):
        self.volume_types = volume_dtypes or dict()
        self.safe = safe

    def process(self, batch, request):
        for volume_type in self.volume_types:
            try:
                volume = batch.volumes[volume_type]
            except KeyError:
                continue
            desired_dtype = self.volume_types[volume_type]
            if volume.data.dtype is not desired_dtype:
                type_casted = volume.data.astype(desired_dtype)
                if self.safe:
                    np.testing.assert_array_equal(volume.data, type_casted)
                volume.data = type_casted
                logger.debug("Casting volume type {0} to {1}".format(volume_type, desired_dtype))
