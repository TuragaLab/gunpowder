import numpy as np

from .provider_test import ProviderTest
from gunpowder import build, VolumeTypes, Typecast


class TestTypecast(ProviderTest):
    def test_typecasts_when_dtype_doesnt_match(self):
        original_dtype = np.dtype("float32")
        new_dtype = np.dtype("uint8")
        # ...
        # assert batch.data.dtype is new_dtype
        # assert batch.data.dtype is not original_dtype
        pass

    def test_doesnt_do_anything_if_dtype_is_already_correct(self):
        # original_id = id(batch.data)
        # original_values = batch.data.copy()
        # do stuff
        # assert id(batch.data) == original_id
        # np.testing.assert_array_equal(batch.data, original_values)
        pass

    def test_doesnt_complain_if_given_a_volume_type_not_in_the_pipeline(self):
        pipeline = self.test_source
        with build(pipeline):
            normal_batch = pipeline.request_batch(self.test_request)
        volume_dtypes = {VolumeTypes.NONEXISTENT_VOLUME_TYPE: np.dtype("bool")}
        pipeline = self.test_source + Typecast(volume_dtypes)
        with build(pipeline):
            batch = pipeline.request_batch(self.test_request)
        self.assertEqual(batch, normal_batch)
