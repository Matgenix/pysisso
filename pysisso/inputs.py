# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


from monty.json import MSONable


class SISSOIN(MSONable):
    """
    Main class containing the input variables for SISSO.
    """

    def __init__(self, ptype=1, ntask=1, nsample=5, task_weighting=1, desc_dim=2, restart=False,
                 nsf=3, rung=2, opset='(+)(-)...', maxcomplexity=10, dimclass='(1:2)(3:3)',
                 maxfval_lb=1e-3, maxfval_ub=1e5, subs_sis=20,
                 method='L0', L1L0_size4L0=1, fit_intercept=True, metric='RMSE', nm_output=100,
                 isconvex=None, width=None):

        pass

# Regression
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ! Keywords for the target properties
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ptype=1              ! property type 1: continuous for regression,2:categorical for classification
# ntask=1              ! number of tasks (properties or maps) 1: single-task learning, >1: multi-task learning
# nsample=5            ! number of samples. If ntask>1, input a number for each task seperated by comma
# task_weighting=1     ! 1: no weighting (tasks treated equally) 2: weighted by #sample_task_i/total_sample.a
# desc_dim=2           ! dimension of the descriptor
# restart=.false.      ! set .true. to continue an unfinished job
#
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ! Keywords for feature construction (FC) and sure independence screening (SIS)
# ! FC recursively do H*Phi->Phi, where H: operators set, Phi: feature space. Number of repeat: rung of the Phi.
# ! Implemented operators:(+)(-)(*)(/)(exp)(exp-)(^-1)(^2)(^3)(sqrt)(cbrt)(log)(|-|)(scd)(^6)(sin)(cos)
# ! scd: standard Cauchy distribution
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# nsf= 3               ! number of input scalar features in the train.dat file
# rung=2               ! rung of the feature space to be constructed
# opset='(+)(-)...'    ! SAME one set of operators for recursive FC. Otherwise, input a set for each rung seperated by comma
# maxcomplexity=10     ! max feature complexity (number of operators in a feature)
# dimclass=(1:2)(3:3)  ! group features according to their dimension/unit; those not in any () are dimensionless
# maxfval_lb=1e-3      ! features having the max. abs. data value <maxfval_lb will not be selected
# maxfval_ub=1e5       ! features having the max. abs. data value >maxfval_ub will not be selected
# subs_sis=20          ! SAME one size for every SIS-selected subspace. Otherwise,input a size for each dimension seperated by comma
#
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ! keywords for descriptor identification via a sparsifying operator
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# method='L0'          ! sparsification operator: 'L1L0' or 'L0'; L0 is recommended!
# L1L0_size4L0= 1      ! If method='L1L0', specify the number of features to be screened by L1 for L0
# fit_intercept=.true. ! fit to a nonzero intercept (.true.) or force the intercept to zero (.false.)
# metric='RMSE'        ! for regression only, the metric for model selection: RMSE,MaxAE
# nm_output=100        ! number of the best models to output

# Classification
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ! Keywords for the target properties
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ptype=2              ! property type 1: continuous for regression,2:categorical for classification
# ntask=1              ! number of tasks (properties or maps) 1: single-task learning, >1: multi-task learning
# nsample=(2,3)        ! (#data in group1,#data in group2,...). If ntask>1, seperate the brackets(tasks) by comma
# desc_dim=2           ! dimension of the descriptor (<=3 for classification,3 is not stable)
# restart=.false.      ! set .true. to continue an unfinished job
#
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ! Keywords for feature construction (FC) and sure independence screening (SIS)
# ! FC recursively do H*Phi->Phi, where H: operators set, Phi: feature space. Number of repeat: rung of the Phi.
# ! Implemented operators:(+)(-)(*)(/)(exp)(exp-)(^-1)(^2)(^3)(sqrt)(cbrt)(log)(|-|)(scd)(^6)(sin)(cos)
# ! scd: standard Cauchy distribution
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# nsf= 3               ! number of input scalar features in the train.dat file
# rung=2               ! rung of the feature space to be constructed
# opset='(+)(-)...'    ! SAME one set of operators for recursive FC. Otherwise, input a set for each rung seperated by comma
# maxcomplexity=10     ! max feature complexity (number of operators in a feature)
# dimclass=(1:2)(3:3)  ! group features according to their dimension/unit; those not in any () are dimensionless
# maxfval_lb=1e-3      ! features having the max. abs. data value <maxfval_lb will not be selected
# maxfval_ub=1e5       ! features having the max. abs. data value >maxfval_ub will not be selected
# subs_sis=20          ! SAME one size for every SIS-selected subspace. Otherwise,input a size for each dimension seperated by comma
#
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ! Keywords for descriptor identification via a sparsifying operator
# !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# method='L0'          ! sparsification method: L0
# isconvex=(1,1)       ! are the domains convex ? 1: YES; 0: NO
# width=0.001          ! boundary tolerance for classification (count in outside points very close to the domain)
# nm_output=100        ! the best candidate models to output


class TrainDat(MSONable):
    """Main class containing the training data for SISSO.
    """
    pass
