![DynamicalSystems.jl logo: The Double Pendulum](https://i.imgur.com/nFQFdB0.gif)

Repository for predicting timeseries using methods from nonlinear dynamics and
timeseries analysis. It
is part of the library [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/latest/).

| **Documentation**   |  **Travis**     | **AppVeyor** | Gitter |
|:--------:|:-------------------:|:-----:|:-----:|
|[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaDynamics.github.io/DynamicalSystems.jl/latest) | [![Build Status](https://travis-ci.org/JuliaDynamics/TimeseriesPrediction.jl.svg?branch=master)](https://travis-ci.org/JuliaDynamics/TimeseriesPrediction.jl) | [![Build status](https://ci.appveyor.com/api/projects/status/amgkws9l1cng2aov?svg=true)](https://ci.appveyor.com/project/JuliaDynamics/timeseriesprediction-jl) | [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/JuliaDynamics/Lobby)

All exported names have detailed documentation strings!

## Spatio-Temporal Timeseries Example
Running the file `examples/barkley_tspred.jl` takes a couple of minutes, but
produces:

![Barkley prediction](https://i.imgur.com/KjnaWIA.gif)

The example first simulates the Barkley model.
Then, after a simulation of `1000` time steps,
it makes a temporal prediction of one of the two fields (`v`) for `200` time steps.
The animated figure shows the true evolution of the model, the prediction
and the error of the prediction.
