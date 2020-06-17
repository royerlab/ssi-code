#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD 2-clause Simplified
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source ssi must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from math import sqrt

import numpy as np

from tv_restoration.convo_operators import ConvolutionOperator
from tv_restoration.image_operators import gradient, norm2sq, div, proj_l2, norm1

VERBOSE = 1


# ----

def power_method(P, PT, data, n_it=10):
    '''
    Calculates the norm of operator K = [grad, P],
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad) + P^T*P :
        ||K|| = sqrt(lambda_max(K^T*K))

    P : forward projection
    PT : back projection
    data : acquired sinogram
    '''
    x = PT(data)
    for k in range(0, n_it):
        x = PT(P(x)) - div(gradient(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)


def chambolle_pock(P, PT, data, Lambda, L, n_it, return_energy=True):
    '''
    Chambolle-Pock algorithm for the minimization of the objective function
        ||P*x - d||_2^2 + Lambda*TV(x)

    P : projection operator
    PT : backprojection operator
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] (see power_method)
    n_it : number of iterations
    return_energy: if True, an array containing the values of the objective function will be returned
    '''

    sigma = 1.0 / L
    tau = 1.0 / L

    x = 0 * PT(data)
    p = 0 * gradient(x)
    q = 0 * data
    x_tilde = 0 * x
    theta = 1.0

    if return_energy: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        p = proj_l2(p + sigma * gradient(x_tilde), Lambda)
        q = (q + sigma * P(x_tilde) - sigma * data) / (1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x + tau * div(p) - tau * PT(q)
        x_tilde = x + theta * (x - x_old)
        # Calculate norms
        if return_energy:
            fidelity = 0.5 * norm2sq(P(x) - data)
            tv = norm1(gradient(x))
            energy = 1.0 * fidelity + Lambda * tv
            en[k] = energy
            if (VERBOSE and k % 10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e" % (k, energy, fidelity, tv))
    if return_energy:
        return en, x
    else:
        return x


def cp_restoration(image, kernel, num_iterations=2500, beta=1e-3):
    # Preparation:
    K = ConvolutionOperator(kernel)
    P = lambda x: K * x
    PT = lambda x: K.T() * x

    # Optimisation:
    L = power_method(P, PT, image, n_it=200)
    print("||K|| = %f" % L)
    en, restored_image = chambolle_pock(P, PT, image, beta, L, num_iterations)

    return restored_image
