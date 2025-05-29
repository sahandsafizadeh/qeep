package model

/*
The Model connects everything that already exists in component and tensor trees using a contract-based approach.
This results in a framework that allows deep neural networks to be defined in a declarative manner and then trained and evaluated effectively.
The contracts that the Model relies on are:
(1) Layer
(2) WeightedLayer
(3) Loss
(4) Metric
(5) Optimizer
(6) BatchGenerator
Each of these is represented by an interface.

By design:
- The Model is built on top of existing concrete implementations in the tensor and component trees. As such, anything it does can be done in other ways.
- The "state" of interfaces (e.g., whether they are nil) is not validated. This means that passing nil interfaces is considered a developer error and may result in a panic.
- The "behavior" and business logic of the interfaces is validated to ensure correct and expected functionality.
*/
