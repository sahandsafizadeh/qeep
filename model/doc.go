package model

/*
Model wires up everything that already exists in component and tensor trees using a contract-based approach.
The result, is a framework that enables us to define deep neural networks in a rather declarative way; then train and evaluate them.
The contracts that model follows are: (1) Layer, (2) WeightedLayer, (3) Loss, (4) Metric, (5) Optimizer and (6) BatchGenerator; each represented by an interface.

By design:
- Model creates a framework on everything concrete that already exists in tensor and component trees; therefore, everything it does can be done in other ways.
- "State" of interfaces are not validated; for instance, their nilness. Thus, for the unpredictable behavior, nil interfaces are taken to be developer's mistake that leads to panic.
- "Behavior" and business logic of interfaces is validated.
*/
