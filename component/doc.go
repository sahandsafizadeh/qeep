package component

/*
Components are separate computational functionalities of a neural network.
They are categorized into 5 major packages:
(1) initializers: for initializing network weights.
(2) layers: for building up a neural network and passing tensors through.
(3) losses: for back-propagation.
(4) metrics: for evaluating the model.
(5) optimizers: for updating network weights.

By design:
- A layer must have one and only one output for any number of inputs.
- Batch dimension must be considered in all components.
*/
