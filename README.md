Repository for predicting timeseries using methods from nonlinear dynamics and
timeseries analysis. It uses [`DelayEmbeddings`](https://github.com/JuliaDynamics/DelayEmbeddings.jl).

| **Documentation**   |  **Travis**     | **AppVeyor** | Gitter |
|:--------:|:-------------------:|:-----:|:-----:|
|[pending... | [![Build Status](https://travis-ci.org/JuliaDynamics/TimeseriesPrediction.jl.svg?branch=master)](https://travis-ci.org/JuliaDynamics/TimeseriesPrediction.jl) | [![Build status](https://ci.appveyor.com/api/projects/status/amgkws9l1cng2aov?svg=true)](https://ci.appveyor.com/project/JuliaDynamics/timeseriesprediction-jl) | [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/JuliaDynamics/Lobby)

All exported names have detailed documentation strings!

## Spatio-Temporal Timeseries Prediction
### Barkley
![Barkley prediction](https://i.imgur.com/LrjrbiS.gif)

The example performs a temporal prediction of the Barkley model.
A simulation of `1000` time steps is given to `temporalprediction`, using the field `v`.
The function attempts to predict for `200` time steps.
The animated figure shows the true evolution of the model, the prediction
and the error of the prediction.

### Kuramoto-Sivashinsky

![Kuramoto-Sivashinsky Prediction](https://i.imgur.com/yDw9UcL.gif)

This example performs a temporal prediction of the Kuramoto-Sivashinsky
model. It is a one-dimensional system with the spatial dimension
shown on the x-axis and its temporal evolution along the y-axis.
The algorithm makes iterative predictions into the future that stay
similar to the true evolution for a while but eventually diverge.
