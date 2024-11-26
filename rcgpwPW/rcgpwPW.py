

from typing import Optional

import tensorflow as tf
import numpy as np
from check_shapes import check_shapes, inherit_check_shapes
from typing import Optional, Tuple, Type, Union, cast
from scipy.stats import chi2


import gpflow

from gpflow.base import InputData, MeanAndVariance, RegressionData, TensorData
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.config import default_float, default_jitter
from gpflow.utilities import to_default_float
from gpflow import posteriors
from gpflow.utilities.ops import eye


from rcgp.utils import add_likelihood_noise_cov, add_noise_cov


    

class RCGPR_with_pursuit(GPModel, InternalDataTrainingLossMixin):
    """
    Robust and Conjugate Gaussian Process Regression with Pursuit Weights
    Extends the RCGPR_deprecated class to include pursuit weighting functionality.
    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "noise_variance: []",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        weighting_function,
        pw=None,  # Optional pursuit weights
        mean_function: Optional[MeanFunction] = None,
        noise_variance: Optional[TensorData] = None,
        likelihood: Optional[Gaussian] = None,
    ):
        """
        Initialize the model with pursuit weights.

        Args:
            data: (X, Y) training data.
            kernel: Kernel function.
            weighting_function: Function for computing weights.
            pw: Optional precomputed pursuit weights. If None, they are calculated automatically.
            mean_function: Optional mean function.
            noise_variance: Optional noise variance.
            likelihood: Optional likelihood.
        """
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)

        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)
        self.weighting_function = weighting_function
        
        dtype = self.data[0].dtype
        self.pw = (
            self.calculate_pursuit_weights(data[0]) if pw is None else tf.convert_to_tensor(pw, dtype=dtype)
        )

    
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.loo_cv()

    @check_shapes(
        "return: []",
    )
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log pseudo smarginal likelihood.

        """
        X, Y = self.data
        n = tf.shape(X)[0]
        K = self.kernel(X)
        W = self.weighting_function.W(X, Y)
        W_dy = self.weighting_function.dy(X, Y)

        K_plus_sW = add_likelihood_noise_cov(K, W, self.likelihood, X)
        L_plus_sW = tf.linalg.cholesky(K_plus_sW + tf.eye(n, dtype=default_float()) * 1e-04)

        nu = (self.likelihood.variance_at(X)**-1)*Y*(W**2)-2*W*W_dy

        A = tf.linalg.triangular_solve(L_plus_sW, tf.transpose(tf.linalg.matmul(nu, K, transpose_a=True)), lower=True)
        B = tf.linalg.triangular_solve(L_plus_sW, nu*(W**-2)*self.likelihood.variance_at(X), lower=True)

        C1 = tf.matmul(Y, Y*(W**2)*self.likelihood.variance_at(X)**-1, transpose_a=True)/2
        C2 = - tf.matmul(Y, W*W_dy, transpose_a=True)
        C3 = - tf.matmul(W, W, transpose_a=True)

        C = tf.reduce_sum(C1+C2+C3)

        D1 = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_plus_sW)))
        D2 = tf.reduce_sum(tf.math.log((W**2)*self.likelihood.variance_at(X)**-1))/2

        D = tf.reduce_sum(D1+D2) 

        return tf.reduce_sum(tf.linalg.matmul(A, B, transpose_a=True))/2 - C - D   
    
        
    # @inherit_check_shapes
    def calculate_median(self, values):
        """
        Calculate the median of a 1D tensor.
        """
        values = tf.sort(values)
        length = tf.shape(values)[0]
        mid = length // 2
        if tf.equal(length % 2, 0):  # Even number of elements
            median = (values[mid - 1] + values[mid]) / 2.0
        else:  # Odd number of elements
            median = values[mid]
        return median
        
    # @inherit_check_shapes
    def projectionstatistics(self, H):
        """
        Compute projection statistics similar to the MATLAB implementation.

        Args:
            H: A TensorFlow tensor of shape [m, n].

        Returns:
            PS: A TensorFlow tensor of shape [m,] containing the maximum projection statistic.
        """
        H = tf.cast(H, tf.float64)  # Ensure tensor is float32
        m, n = tf.shape(H)[0], tf.shape(H)[1]

        # Compute median along each column
        M = tf.map_fn(self.calculate_median, tf.transpose(H))
        M = tf.transpose(M)

        # Initialize the projection statistics matrix
        P = tf.zeros((m, m), dtype=tf.float64)

        for kk in range(m):
            # Center the kk-th row around the median
            u_kk = H[kk, :] - M
            norm_u_kk = tf.linalg.norm(u_kk)  # Norm of the centered row

            # Avoid division by zero in normalization
            norm_u_kk = tf.where(tf.equal(norm_u_kk, 0), 1e-8, norm_u_kk)
            v_kk = u_kk / norm_u_kk  # Normalize the centered row

            # Projection of all rows onto the direction of v_kk
            z = tf.linalg.matvec(H, v_kk)

            # Compute the median of projections
            zmed = self.calculate_median(z)

            # Compute MAD (Median Absolute Deviation)
            mad = tf.maximum(
                1.4826 * (1 + 15 / tf.cast(m, tf.float64)) * self.calculate_median(tf.abs(z - zmed)),
                1e-8,
            )  # Avoid division by zero for MAD

            # Compute projection statistics for all rows
            P_row = tf.abs(z - zmed) / mad
            P = tf.tensor_scatter_nd_update(P, [[kk]], [P_row])

        # Compute the maximum projection statistic for each row
        PS = tf.reduce_max(P, axis=0)

        return PS
        
    # @inherit_check_shapes
    def calculate_pursuit_weights(self, X, cutoff_scale=0.975):
        """
        Calculate pursuit weights for TensorFlow tensors based on projection statistics.

        Args:
            X: A TensorFlow tensor of shape [m, n].
            cutoff_scale: Cutoff scale for chi-squared distribution.

        Returns:
            pw: A TensorFlow tensor of shape [m, 1] containing pursuit weights.
        """
        X = tf.cast(X, tf.float64)  # Ensure X is float32
        ps = self.projectionstatistics(X)  # Compute projection statistics
        cutoff = chi2.ppf(cutoff_scale, df=int(X.shape[1]))  # Chi-squared cutoff
        cutoff = tf.constant(cutoff, dtype=tf.float64)  # Ensure dtype compatibility
        pw = tf.minimum(1.0, cutoff / tf.square(ps))  # Calculate weights
        # print("PW:", pw)
        return tf.expand_dims(pw, axis=-1)
        

    # @inherit_check_shapes
    def loo_cv(self) -> tf.Tensor:
        """
        Computes the leave-one-out cross-validation (LOO-CV) loss for training.
        Adjusts residuals using pursuit weights (pw).
        """
        X, Y = self.data
        mean_output = tf.cast(self.mean_function(X), Y.dtype)
        err = (Y - mean_output) / self.pw  # Adjust residuals with pursuit weights
        K = self.kernel(X)
        n = tf.cast(tf.shape(X)[0], K.dtype)
        likelihood_variance = self.likelihood.variance_at(X)
        W, W_dy = self.weighting_function.w_dy(X, err)
        dylog2 = 2 * likelihood_variance * W_dy / W
        Y_bar = err - dylog2

        K_sW = add_noise_cov(K, tf.squeeze(W, axis=-1), tf.squeeze(likelihood_variance, axis=-1))
        L_sW = tf.linalg.cholesky(K_sW)
        L_sW_inv = tf.linalg.inv(L_sW)
        diag_K_sW_inv = tf.reshape(tf.reduce_sum(L_sW_inv ** 2, axis=0), (-1, 1))

        A = diag_K_sW_inv * dylog2
        B = tf.matmul(L_sW_inv, tf.matmul(L_sW_inv, Y_bar), transpose_a=True)
        C = diag_K_sW_inv * (1 - diag_K_sW_inv * (likelihood_variance * (W**-2) - likelihood_variance))
        D = C / diag_K_sW_inv**2

        loo = -0.5 * tf.reduce_sum(tf.math.log(D))
        loo -= 0.5 * n * np.log(2 * np.pi)
        loo -= 0.5 * tf.reduce_sum((A + B) ** 2 / C)
        return loo
#
    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Make predictions with the pursuit weights affecting the residuals.
        """
        X, Y = self.data
        mean_output = tf.cast(self.mean_function(X), Y.dtype)
        err = (Y - mean_output) / self.pw  # Adjust residuals with pursuit weights
        m = tf.shape(X)[0]

        W = self.weighting_function.W(X, err)
        W_dy = self.weighting_function.dy(X, err)
        likelihood_variance = self.likelihood.variance_at(X)

        kmm = self.kernel(X)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X, Xnew)
        kmm_plus_sW = add_likelihood_noise_cov(kmm, W, self.likelihood, X)

        Lm_plus_sW = tf.linalg.cholesky(
            kmm_plus_sW + tf.eye(m, dtype=default_float()) * 1e-06
        )

        K = tf.rank(kmn)
        perm = tf.concat(
            [
                tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
                tf.reshape(0, [1]),  # [W]
                tf.reshape(K - 1, [1]),
            ],
            0,
        )  # [N]
        kmn = tf.transpose(kmn, perm)  # [..., W, N]

        leading_dims = tf.shape(kmn)[:-2]

        # Compute the projection matrix A
        Lm_plus_sW = tf.broadcast_to(Lm_plus_sW, tf.concat([leading_dims, tf.shape(Lm_plus_sW)], 0))  # [..., W, W]
        A = tf.linalg.triangular_solve(Lm_plus_sW, kmn, lower=True)  # [..., W, N]

        # Compute kernel stuff
        num_func = tf.shape(err)[-1]  # R
        N = tf.shape(kmn)[-1]

        if full_cov:
            f_var = knn - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]
            cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
            f_var = tf.broadcast_to(tf.expand_dims(f_var, -3), cov_shape)  # [..., R, N, N]
        else:
            f_var = knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]
            cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
            f_var = tf.broadcast_to(tf.expand_dims(f_var, -2), cov_shape)  # [..., R, N]

        if not full_cov:
            f_var = tf.linalg.adjoint(f_var)  # [N, R]

        f = err - 2 * likelihood_variance * W_dy / W
        B = tf.linalg.triangular_solve(Lm_plus_sW, f, lower=True)  # [..., W, N]
        f_mean_zero = tf.linalg.matmul(A, B, transpose_a=True)

        f_mean = f_mean_zero + self.mean_function(Xnew)

        return f_mean, f_var
        
   

class RCGPR_with_posterior(RCGPR_with_pursuit):
    """
    This is an implementation of GPR that provides a posterior() method that
    enables caching for faster subsequent predictions.
    """

    def posterior(
        self,
        precompute_cache: posteriors.PrecomputeCacheType = posteriors.PrecomputeCacheType.TENSOR,
    ):
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """

        return RCGPPosterior(
            kernel=self.kernel,
            data=self.data,
            weighting_function=self.weighting_function,
            pw=self.pw,
            likelihood=self.likelihood,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        For backwards compatibility, GPR's predict_f uses the fused (no-cache)
        computation, which is more efficient during training.

        For faster (cached) prediction, predict directly from the posterior object, i.e.,:
            model.posterior().predict_f(Xnew, ...)
        """
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )


class RCGPPosterior(posteriors.AbstractPosterior):
    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, Q]",
    )
    def __init__(
        self,
        kernel: Kernel,
        data: RegressionData,
        weighting_function,
        pw: tf.Tensor,
        likelihood: Gaussian,
        mean_function: MeanFunction,
        *,
        precompute_cache: Optional[posteriors.PrecomputeCacheType],
    ) -> None:
        X, Y = data
        super().__init__(kernel, X, mean_function=mean_function)
        self.Y_data = Y
        self.likelihood = likelihood
        self.weighting_function = weighting_function
        self.pw=pw

        if precompute_cache is not None:
            self.update_cache(precompute_cache)

    @inherit_check_shapes
    def _conditional_with_precompute(
        self,
        cache: Tuple[tf.Tensor, ...],
        Xnew,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function.
        Relies on cached alpha and Qinv.
        """
        err, Lm_plus_sW, W, W_dy, likelihood_variance = cache
        err= err / self.pw

        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(self.X_data, Xnew)


        K = tf.rank(kmn)
        perm = tf.concat(
            [
                tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
                tf.reshape(0, [1]),  # [W]
                tf.reshape(K - 1, [1]),
            ],
            0,
        )  # [N]
        kmn = tf.transpose(kmn, perm)  # [..., W, N]

        leading_dims = tf.shape(kmn)[:-2]
        M = tf.shape(err)[-2]

        # Compute the projection matrix A
        Lm_plus_sW = tf.broadcast_to(Lm_plus_sW, tf.concat([leading_dims, tf.shape(Lm_plus_sW)], 0))  # [..., W, W]
        A = tf.linalg.triangular_solve(Lm_plus_sW, kmn, lower=True)  # [..., W, N]

        # compute kernel stuff
        num_func = tf.shape(err)[-1]  # R
        N = tf.shape(kmn)[-1]

        if full_cov:
            f_var = knn - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]
            cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
            f_var = tf.broadcast_to(tf.expand_dims(f_var, -3), cov_shape)  # [..., R, N, N]
        else:
            f_var = knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]
            cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
            f_var = tf.broadcast_to(tf.expand_dims(f_var, -2), cov_shape)  # [..., R, N]

        if not full_cov:
            f_var = tf.linalg.adjoint(f_var)  # [N, R]

        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm_plus_sW), A, lower=False)
  
        f = err - 2*likelihood_variance*W_dy/W
        f_shape = tf.concat([leading_dims, [M, num_func]], 0)  # [..., M, R]
        f = tf.broadcast_to(f, f_shape)

        f_mean = tf.linalg.matmul(A, f, transpose_a=True)

        return f_mean, f_var

    def _precompute(self) -> Tuple[posteriors.PrecomputedValue, ...]:
        assert self.mean_function is not None
        X_data = cast(tf.Tensor, self.X_data)
        err = self.Y_data - self.mean_function(X_data)
        err= err / self.pw
        
        D = err.shape[1]
        M = X_data.shape[0]
        D_dynamic = D is None
        M_dynamic = M is None

        W = self.weighting_function.W(X_data, err)
        W_dy = self.weighting_function.dy(X_data, err)

        likelihood_variance = self.likelihood.variance_at(self.X_data)

        kmm = self.kernel(X_data)
        kmm_plus_sW = add_likelihood_noise_cov(kmm, W, self.likelihood, X_data) + eye(
            tf.shape(X_data)[-2], value=default_jitter(), dtype=default_float()
        )

        Lm_plus_sW = tf.linalg.cholesky(kmm_plus_sW)

        return (
            posteriors.PrecomputedValue(err, (M_dynamic, D_dynamic)),
            posteriors.PrecomputedValue(Lm_plus_sW, (M_dynamic, M_dynamic)),
            posteriors.PrecomputedValue(W, (M_dynamic, D_dynamic)),
            posteriors.PrecomputedValue(W_dy, (M_dynamic, D_dynamic)),
            posteriors.PrecomputedValue(likelihood_variance, (M_dynamic, D_dynamic)),
        )

    @inherit_check_shapes
    def _conditional_fused(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function
        Does not make use of caching
        """
        temp_cache = tuple(c.value for c in self._precompute())
        return self._conditional_with_precompute(temp_cache, Xnew, full_cov, full_output_cov)


# class RCGPR_with_pursuit(RCGPR_deprecated):   #this is pursuit weighting [added]
#     """
#     Robust and Conjugate Gaussian Process Regression with Pursuit Weights.
#     Extends RCGPR_deprecated to include pursuit weighting functionality.
#     """
#     def __init__(self, data, kernel, weighting_function, pw=None, **kwargs):
#         """
#         Args:
#             data: RegressionData - (X, Y) training data.
#             kernel: Kernel - Kernel function.
#             weighting_function: Weighting function (e.g., IMQ).
#             pw: Optional[Tensor] - Precomputed pursuit weights; if None, they will be calculated.
#         """
#         super().__init__(data, kernel, weighting_function, **kwargs)
#         self.pw = calculate_pursuit_weights(data[0]) if pw is None else pw

#     def loo_cv(self) -> tf.Tensor:
#         """
#         Computes the leave-one-out cross-validation (LOO-CV) loss for training.
#         Includes pursuit weights (pw) to adjust residuals.
#         """
#         X, Y = self.data
#         err = (Y - self.mean_function(X)) / self.pw  # Adjust residuals with pursuit weights
#         print(self.pw)
#         K = self.kernel(X)
#         n = tf.cast(tf.shape(X)[0], K.dtype)
#         likelihood_variance = self.likelihood.variance_at(X)
#         W, W_dy = self.weighting_function.w_dy(X, err)
#         dylog2 = 2 * likelihood_variance * W_dy / W
#         Y_bar = err - dylog2

#         K_sW = add_noise_cov(K, tf.squeeze(W, axis=-1), tf.squeeze(likelihood_variance, axis=-1))
#         L_sW = tf.linalg.cholesky(K_sW)
#         diag_K_sW_inv = tf.reshape(tf.reduce_sum(L_sW_inv ** 2, axis=0), (-1, 1))

#         A = diag_K_sW_inv * dylog2
#         B = tf.matmul(L_sW_inv, tf.matmul(L_sW_inv, Y_bar), transpose_a=True)
#         C = diag_K_sW_inv * (1 - diag_K_sW_inv * (likelihood_variance * (W**-2) - likelihood_variance))
#         D = C / diag_K_sW_inv**2

#         loo = -0.5 * tf.reduce_sum(tf.math.log(D))
#         loo -= 0.5 * n * np.log(2 * np.pi)
#         loo -= 0.5 * tf.reduce_sum((A + B) ** 2 / C)
#         return loo


class RCGPRWPW(RCGPR_with_posterior):
    __doc__ = RCGPR_with_pursuit.__doc__ 
