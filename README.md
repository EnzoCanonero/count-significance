This repo contains code to perform to taks:
-   Compute Asimov median significance for test statistics r and r*, and compare them with exact numerical
-   Copmute single p-value prediction for given observed data, again comparing r and r* predicitons with exact numerical prediction

This is done for two model:
- Simple counting, "On", experiment
- "On-off" problem

The folder "parallelization" is used for to perform the medsig computation for the on-off problem in a parallelize manner, if condor is available.
