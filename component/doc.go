package component

/*
Components are modular computational building blocks of a neural network.
They are categorized into five main packages:
(1) initializers: for initializing network weights.
(2) layers: for building up a neural network and passing tensors through.
(3) losses: for back-propagation.
(4) metrics: for evaluating model performance.
(5) optimizers: for updating network weights.

By design:
- Each layer must have exactly one output, regardless of the number of inputs.
- All components must consider the batch dimension.
- A component is responsible for managing its tensors' device.
*/
