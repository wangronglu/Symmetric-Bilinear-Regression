# Symmetric-Bilinear-Regression
Matlab code for fitting a symmetric bilinear model:

<a href="https://www.codecogs.com/eqnedit.php?latex=E(y\mid&space;W)&space;=&space;\alpha&plus;\sum_{h=1}^{K}\lambda_{h}\boldsymbol{\beta}_{h}^{\top}W\boldsymbol{\beta}_{h}." target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(y\mid&space;W)&space;=&space;\alpha&plus;\sum_{h=1}^{K}\lambda_{h}\boldsymbol{\beta}_{h}^{\top}W\boldsymbol{\beta}_{h}." title="E(y\mid W) = \alpha+\sum_{h=1}^{K}\lambda_{h}\boldsymbol{\beta}_{h}^{\top}W\boldsymbol{\beta}_{h}" /></a>

with L1 penalty on entries of
<a href="https://www.codecogs.com/eqnedit.php?latex=\{\lambda_{h}\boldsymbol{\beta}_{h}\boldsymbol{\beta}_{h}^{\top}\}_{h=1}^{K}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\{\lambda_{h}\boldsymbol{\beta}_{h}\boldsymbol{\beta}_{h}^{\top}\}_{h=1}^{K}" title="\{\lambda_{h}\boldsymbol{\beta}_{h}\boldsymbol{\beta}_{h}^{\top}\}_{h=1}^{K}" /></a>.

The parameters of the symmetric bilinear model, 
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha,\{\lambda_{h}\},\{\boldsymbol{\beta}_{h}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha,\{\lambda_{h}\},\{\boldsymbol{\beta}_{h}\}" title="\alpha,\{\lambda_{h}\},\{\boldsymbol{\beta}_{h}\}" /></a>,
are estimated by solving the following optimization

<a href="https://www.codecogs.com/eqnedit.php?latex=\underset{\alpha,\{\lambda_{h}\},\{\boldsymbol{\beta}_{h}\}}{\min}\&space;\dfrac{1}{2n}\sum_{i=1}^{n}\left(y_{i}-\alpha-\sum_{h=1}^{K}\lambda_{h}\boldsymbol{\beta}_{h}^{\top}W_{i}\boldsymbol{\beta}_{h}\right)^{2}&plus;\gamma\sum_{h=1}^{K}\left|\lambda_{h}\right|\sum_{u=1}^{V}\sum_{v<u}\left|\beta_{hu}\beta_{hv}\right|.&space;\label{eq:SBL_optimization}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{\alpha,\{\lambda_{h}\},\{\boldsymbol{\beta}_{h}\}}{\min}\&space;\dfrac{1}{2n}\sum_{i=1}^{n}\left(y_{i}-\alpha-\sum_{h=1}^{K}\lambda_{h}\boldsymbol{\beta}_{h}^{\top}W_{i}\boldsymbol{\beta}_{h}\right)^{2}&plus;\gamma\sum_{h=1}^{K}\left|\lambda_{h}\right|\sum_{u=1}^{V}\sum_{v<u}\left|\beta_{hu}\beta_{hv}\right|.&space;\label{eq:SBL_optimization}" title="\underset{\alpha,\{\lambda_{h}\},\{\boldsymbol{\beta}_{h}\}}{\min}\ \dfrac{1}{2n}\sum_{i=1}^{n}\left(y_{i}-\alpha-\sum_{h=1}^{K}\lambda_{h}\boldsymbol{\beta}_{h}^{\top}W_{i}\boldsymbol{\beta}_{h}\right)^{2}+\gamma\sum_{h=1}^{K}\left|\lambda_{h}\right|\sum_{u=1}^{V}\sum_{v<u}\left|\beta_{hu}\beta_{hv}\right|." /></a>

