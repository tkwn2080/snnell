I want to teach a false thing to smell. This has been built to that end.

The first iteration was a simulation of particle dynamics, an odour particle emitter on the right.
The entity, initially a small green circle with two antennae, was initialised on the left.
The entity had various settings: probe angle, response angle, movement speed, distance, etc.
It would move at the response angle, based on the movement speed and distance, when the probe collided with an odour particle.
This was evolved by way of a simple genetic algorithm, initially asexual reproduction and then sexual reproduction.
This worked really well, it was almost eery to see this blind thing move with apparent intelligence.

The second iteration gave the creature a brain, which initially made it much stupider.
I have here built a simple spiking neural network (SNN) implementation based on my conceptual understanding of neurons.
To this I have attempted to add reinforcement learning, and the whole has been an exercise in understanding.
The genetic algorithm aspect was maintained and used to select architectures, etc.

The third iteration, that present, has been a rewrite of the simulation structure.
I have implemented parallelisation in a headless mode so that I can have larger runs.
I am currently working on this, intending to use an evolutionary strategy to tune the networks.
