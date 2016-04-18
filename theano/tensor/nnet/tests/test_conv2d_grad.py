from __future__ import absolute_import, print_function, division
import time

from nose.plugins.skip import SkipTest
import numpy
import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt
from theano.tensor.nnet import conv
from theano.tensor.basic import _allclose, NotScalarConstantError
from theano.tests.unittest_tools import attr
from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs, conv2d_grad_wrt_weights

class TestConv2DgradWrtInputWeight(utt.InferShapeTester):
    # This class contains tests for the  conv2d_grad_wrt_inputs, 
    # conv2d_grad_wrt_weights, but will also be inherited 
    # from for other implementations
    mode = None
    dtype = theano.config.floatX
    # This will be set to the appropriate function in the inherited classes.
    # The call to `staticmethod` is necessary to prevent Python from passing
    # `self` as the first argument.
    conv2d = staticmethod(conv.conv2d)

    def setUp(self):
        super(TestConv2DgradWrtInputWeight, self).setUp()
        self.input = T.tensor4('input', dtype=self.dtype)
        self.input.name = 'default_V'
        self.filters = T.tensor4('filters', dtype=self.dtype)
        self.filters.name = 'default_filters'
        if not conv.imported_scipy_signal and theano.config.cxx == "":
            raise SkipTest("conv2d tests need SciPy or a c++ compiler")

    def validate(self, image_shape, filter_shape,
                 border_mode='valid', subsample=(1, 1),
                 N_image_shape=None, N_filter_shape=None,
                 input=None, filters=None,
                 unroll_batch=None, unroll_kern=None, unroll_patch=None,
                 verify_grad=True, should_raise=False, filter_flip=True):
        """
        :param image_shape: The constant shape info passed to conv2d.
        :param filter_shape: The constant shape info passed to conv2d.

        :param N_image_shape: None(default to image_shape) or tuple of
                              4 elements with the shape of the input image

        :param N_filter_shape: None(default to filter_shape) or tuple
                               of 4 elements with the shape of the
                               input filter

        """
        if N_image_shape is None:
            N_image_shape = [T.get_scalar_constant_value(T.
                as_tensor_variable(x)) for x in image_shape]
        if N_filter_shape is None:
            N_filter_shape = [T.get_scalar_constant_value(T.
                as_tensor_variable(x)) for x in filter_shape]

        if input is None:
            input = self.input
        if not filters:
            filters = self.filters

        ############# THEANO IMPLEMENTATION ############
        def sym_conv2d_grad_wrt_inputs(output_grad, filters):
            
            rval = conv2d_grad_wrt_inputs( output_grad      = output_grad,
                                           filters          = filters,
                                           input_shape      = image_shape,
                                           filter_shape     = filter_shape,
                                           border_mode      = border_mode,
                                           subsample        = subsample,
                                           filter_flip      = filter_flip)
            rval.name = 'conv2d_grad_wrt_inputs_output'
            return rval

        # we create a symbolic function so that verify_grad can work
        def sym_conv2d(input, filters):
            # define theano graph and function
            input.name = 'input'
            filters.name = 'filters'
            rval =  conv.conv2d(input, filters, image_shape, filter_shape,
                          border_mode, subsample, unroll_batch=unroll_batch,
                          unroll_kern=unroll_kern, unroll_patch=unroll_patch)
            rval.name = 'conv_output'
            return rval

        output = sym_conv2d_grad_wrt_inputs(input, filters)
        output.name = 'conv2d_grad_wrt_inputs(%s,%s)' % (input.name, filters.name)
        theano_conv_grad_wrt_input = theano.function([input, filters], output, mode=self.mode)

        # initialize input and compute result
        image_data = numpy.random.random(N_image_shape).astype(self.dtype)
        filter_data = numpy.random.random(N_filter_shape).astype(self.dtype)
        orig_image_data = image_data
        try:
            theano_output = theano_conv_grad_wrt_input(image_data, filter_data)
        except ValueError:
            if not should_raise:
                raise
            return
        else:
            if should_raise:
                raise Exception(
                "ConvOp should have generated an error")

        if verify_grad:
            utt.verify_grad(sym_conv2d_grad_wrt_inputs, [orig_image_data, filter_data])
    
    def test_basic1(self):
        """Tests that basic convolutions work for odd and even
        dimensions of image and filter shapes, as well as rectangular
        images and filters.

        """
        self.validate((2, 2, 3, 3), (2, 2, 2, 2), 'valid', verify_grad=False)

    def test_basic(self):
        """Tests that basic convolutions work for odd and even
        dimensions of image and filter shapes, as well as rectangular
        images and filters.

        """
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), 'valid', verify_grad=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid')
        self.validate((3, 2, 7, 5), (5, 2, 3, 2), 'valid', verify_grad=False)
        self.validate((3, 2, 8, 8), (4, 2, 5, 5), 'full', verify_grad=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full')
        # test filter same size as input

    def test_uint_image_shape_datatype(self):
        """Tests for uint datatype in image_shape.

        """
        self.validate((2, 2, 3, numpy.uint8(3)), (3, 2, 3, 3), 'valid', verify_grad=False)
        self.validate((numpy.uint16(2), 2, 3, 3), (3, 2, 3, 3), 'valid', verify_grad=False)
        self.validate((2, numpy.uint32(2), 3, 3), (3, 2, 3, 3), 'valid', verify_grad=False)
        
    def test_uint_filter_shape_datatype(self):
        """Tests for uint datatype in filter_shape

        """
        self.validate((3, 2, 3, 3), (2, 2, 3, numpy.uint8(3)), 'valid', verify_grad=False)
        self.validate((3, 2, 3, 3), (numpy.uint16(2), 2, 3, 3), 'valid', verify_grad=False)
        self.validate((3, 2, 3, 3), (2, numpy.uint32(2), 3, 3), 'valid', verify_grad=False)
        
    def test_img_kernel_same_shape(self):
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'full')
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid')

    def test_unroll_patch_true(self):
        """
        Test basic convs with True.
        """
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', unroll_patch=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', unroll_patch=True)
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid',
             unroll_patch=True, verify_grad=False)

    def test_unroll_patch_false(self):
        """
        Test basic convs with False.
        """
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', unroll_patch=False)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', unroll_patch=False)
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid',
             unroll_patch=False, verify_grad=False)

    def test_unroll_patch_true_fail(self):
        """
        Test basic convs with True.
        """
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', unroll_patch=True,
                      N_image_shape=(1, 3, 3, 3),  N_filter_shape=(6, 3, 2, 2),
                      should_raise=True)
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', unroll_patch=True,
                      N_image_shape=(1, 3, 3, 3),  N_filter_shape=(6, 3, 2, 2),
                      should_raise=True)
        self.validate((3, 2, 3, 3), (4, 2, 3, 3), 'valid', unroll_patch=True,
                      N_image_shape=(1, 3, 3, 3),  N_filter_shape=(6, 3, 2, 2),
                      should_raise=True)

    def test_unroll_special(self):
        """
        (unroll_kern, unroll_batch) in (0,1),(1,0) is special case.
        """
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid', unroll_batch=1)

    def test_unroll_batch(self):
        """
        Test mini-batch unrolling for various legal values.
        """
        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
             unroll_batch=2, verify_grad=False)
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
             unroll_batch=3, verify_grad=False)

    def test_unroll_kern(self):
        """
        Test kernel unrolling for various legal values.
        """
        # 6 filters is a multiple of 2 and 3. Should work.
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid', unroll_kern=2,
             verify_grad=False)
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid', unroll_kern=3,
             verify_grad=False)

    def test_unroll_batch_kern(self):
        """Test mini-batch unrolling with kernel unrolling for various
        legal values.

        """
        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
             unroll_batch=2, unroll_kern=3, verify_grad=False)
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
             unroll_batch=3, unroll_kern=3, verify_grad=False)
        # 6 filters is a multiple of 2 and 3. Should work.
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid',
             unroll_batch=2, unroll_kern=2, verify_grad=False)
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid',
             unroll_batch=2, unroll_kern=3, verify_grad=False)

    def test_unroll_batch_kern_fail(self):
        """Test mini-batch unrolling with kernel unrolling for various
        legal values, but pass bad input.  All those test must
        generate errors

        """
        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
                      unroll_batch=2, unroll_kern=3,
                      N_image_shape=(7, 2, 3, 3), N_filter_shape=(3, 2, 2, 2),
                      should_raise=True)
        self.validate((6, 2, 3, 3), (3, 2, 2, 2), 'valid',
                      unroll_batch=3, unroll_kern=3,
                      N_image_shape=(6, 2, 3, 3), N_filter_shape=(4, 2, 2, 2),
                      should_raise=True)
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid',
                      unroll_batch=2, unroll_kern=2,
                      N_image_shape=(1, 3, 3, 3),  N_filter_shape=(6, 3, 2, 2),
                      should_raise=True)
        self.validate((2, 3, 3, 3), (6, 3, 2, 2), 'valid',
                      unroll_batch=2, unroll_kern=3,
                      N_image_shape=(2, 3, 3, 3),  N_filter_shape=(5, 3, 2, 2),
                      should_raise=True)

    @attr('slow')
    def test_subsample(self):
        """
        Tests convolution where subsampling != (1,1)
        """
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'full', subsample=(2, 2))
        self.validate((3, 2, 7, 5), (5, 2, 2, 3), 'valid', subsample=(2, 1))
        self.validate((1, 1, 6, 6), (1, 1, 3, 3), 'valid', subsample=(3, 3))

        # Fails as of 2012-07-11
        self.assertRaises(NotImplementedError, self.validate, (1, 1, 6, 6),
                          (1, 1, 3, 3), 'full', subsample=(3, 3))

    def test_shape_Constant_tensor(self):
        """
        Tests convolution where the {image,filter}_shape is a Constant tensor.
        """
        as_t = T.as_tensor_variable
        self.validate((as_t(3), as_t(2), as_t(7), as_t(5)), (5, 2,
             2, 3), 'valid')
        self.validate(as_t([3, 2, 7, 5]), (5, 2, 2, 3), 'valid')
        self.validate(as_t((3, 2, 7, 5)), (5, 2, 2, 3), 'valid')
        self.validate((3, 2, 7, 5), (as_t(5), as_t(2), as_t(2),
             as_t(3)), 'valid')
        self.validate((3, 2, 7, 5), as_t([5, 2, 2, 3]), 'valid')
        self.validate((3, 2, 7, 5), as_t((5, 2, 2, 3)), 'valid')
        self.validate(as_t([3, 2, 7, 5]), as_t([5, 2, 2, 3]), 'full')

    def test_invalid_filter_shape(self):
        """
        Tests scenario where filter_shape[1] != input_shape[1]
        """
        self.assertRaises(AssertionError, self.validate,
                          (3, 2, 8, 8), (4, 3, 5, 5),
                          'valid')

    @attr('slow')
    def test_invalid_input_shape(self):
        """
        Tests that when the shape gived at build time is not the same as
        run time we raise an error
        """
        for unroll_batch in [None, 1, 3]:
            for unroll_kern in [None, 2, 4]:
                for unroll_patch in [None, True, False]:
                    for mode in ['valid', 'full']:
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_image_shape=(2, 2, 8, 8),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_image_shape=(3, 1, 8, 8),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_image_shape=(3, 2, 7, 8),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_image_shape=(3, 2, 8, 7),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)

                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_filter_shape=(3, 2, 5, 5),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_filter_shape=(4, 1, 5, 5),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_filter_shape=(4, 2, 6, 5),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)
                        self.assertRaises(ValueError, self.validate,
                                          (3, 2, 8, 8), (4, 2, 5, 5),
                                          mode, N_filter_shape=(4, 2, 5, 6),
                                          unroll_batch=unroll_batch,
                                          unroll_kern=unroll_kern,
                                          unroll_patch=unroll_patch)

    def test_missing_info(self):
        """
        Test convolutions for various pieces of missing info.
        """
        self.validate(None, None,
                      N_image_shape=(3, 2, 8, 8),
                      N_filter_shape=(4, 2, 5, 5))
        self.validate((3, 2, None, None), None,
                      N_image_shape=(3, 2, 8, 8),
                      N_filter_shape=(4, 2, 5, 5))
        self.validate((None, 2, None, None), (None, 2, 5, 5),
                      N_image_shape=(3, 2, 8, 8),
                      N_filter_shape=(4, 2, 5, 5))
        self.validate((3, 2, 8, 8), (4, 2, None, 5),
                      N_image_shape=(3, 2, 8, 8),
                      N_filter_shape=(4, 2, 5, 5))
        self.validate((3, 2, 8, 8), (4, 2, 5, None),
                      N_image_shape=(3, 2, 8, 8),
                      N_filter_shape=(4, 2, 5, 5))

    def test_wrong_info(self):
        """
        Test convolutions when we don't give a constant as shape information
        """
        i = theano.scalar.basic.int32()
        self.assertRaises(NotScalarConstantError, self.validate,
                          (3, 2, 8, i), (4, 2, 5, 5),
                          N_image_shape=(3, 2, 8, 8),
                          N_filter_shape=(4, 2, 5, 5))
        self.assertRaises(NotScalarConstantError, self.validate,
                          (3, 2, 8, 8), (4, 2, 5, i),
                          N_image_shape=(3, 2, 8, 8),
                          N_filter_shape=(4, 2, 5, 5))

    def test_full_mode(self):
        """
        Tests basic convolution in full mode and case where filter
        is larger than the input image.
        """
        self.validate((3, 2, 5, 5), (4, 2, 8, 8), 'full')

        def f():
            self.validate((3, 2, 5, 5), (4, 2, 8, 8), 'valid')
        self.assertRaises(Exception, f)

    def test_wrong_input(self):
        """
        Make sure errors are raised when image and kernel are not 4D tensors
        """
        self.assertRaises(Exception, self.validate, (3, 2, 8, 8), (4, 2, 5, 5),
                          'valid', input=T.dmatrix())
        self.assertRaises(Exception, self.validate, (3, 2, 8, 8), (4, 2, 5, 5),
                          'valid', filters=T.dvector())
        self.assertRaises(Exception, self.validate, (3, 2, 8, 8), (4, 2, 5, 5),
                          'valid', input=T.dtensor3())

    def test_gcc_crash(self):
        """
        gcc 4.3.0 20080428 (Red Hat 4.3.0-8)

        crashed in this following case. I changed the c code to don't hit
        gcc bug. So it should not crash anymore
        """
        self.validate((1, 10, 213, 129), (46, 10, 212, 1), 'valid',
                      verify_grad=False)


if __name__ == '__main__':

    t = TestConv2DgradWrtInputWeight('setUp')
    t.setUp()
