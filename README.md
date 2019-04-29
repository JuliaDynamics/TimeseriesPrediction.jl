![TimeseriesPredition.jl logo](https://github.com/JuliaDynamics/JuliaDynamics/blob/master/videos/tspred/tspred_logo.png?raw=true)

| **Documentation**   |  **Travis**     | **AppVeyor** | Gitter |
|:--------:|:-------------------:|:-----:|:-----:|
|[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliadynamics.github.io/TimeseriesPrediction.jl/latest) | [![Build Status](https://travis-ci.org/JuliaDynamics/TimeseriesPrediction.jl.svg?branch=master)](https://travis-ci.org/JuliaDynamics/TimeseriesPrediction.jl) | [![Build status](https://ci.appveyor.com/api/projects/status/amgkws9l1cng2aov?svg=true)](https://ci.appveyor.com/project/JuliaDynamics/timeseriesprediction-jl) | [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/JuliaDynamics/Lobby)

Repository for predicting timeseries using methods from nonlinear dynamics and
timeseries analysis. It uses [`DelayEmbeddings`](https://github.com/JuliaDynamics/DelayEmbeddings.jl).

## Kuramoto-Sivashinsky example

![Kuramoto-Sivashinsky Prediction](https://i.imgur.com/yDw9UcL.gif)

This example performs a temporal prediction of the Kuramoto-Sivashinsky
model. It is a one-dimensional system with the spatial dimension
shown on the x-axis and its temporal evolution along the y-axis.
The algorithm makes iterative predictions into the future that stay
similar to the true evolution for a while but eventually diverge.
