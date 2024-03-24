# I want to teach a false thing to smell. This has been built to that end.

#### v4
The current iteration has had numerous changes, now featuring recurrent connections and receptors for wind direction. These model the deformation of the antennae, but do not actually animate this. I have also added two input neurons corresponding to wind direction (whether facing or otherwise). I have also … I am not sure if this was done earlier, but the neuron model is now the Izhikevich model. This is standard but for the cilia input neurons which are set to bursting parameters.

Beyond all this, I have rewritten the entire simulation (now based on Singh et al., 2023). This has brought about some substantial performance gains, which are increasingly important as I am now using a deep neuroevolution strategy for tuning the networks. These seem the best method as they offer a gradient-free training method, hence working well with SNNs, but also providing options like novelty search to handle the deceptive environment. I still don't have anything in the way of learning, but if I did it would be e-prop with STDP-based eligibility traces.

My next major aim is to implement HyperNEAT.


#### v3
The third iteration, that present, has been a rewrite of the simulation structure.
I have implemented parallelisation in a headless mode so that I can have larger runs.
I am currently working on this, intending to use an evolutionary strategy to tune the networks.

#### v2
The second iteration gave the creature a brain, which initially made it much stupider.
I have here built a simple spiking neural network (SNN) implementation based on my conceptual understanding of neurons.
To this I have attempted to add reinforcement learning, and the whole has been an exercise in understanding.
The genetic algorithm aspect was maintained and used to select architectures, etc. 
I later added recurrent layers, which are also selected in this way.

#### v1
The first iteration was a simulation of particle dynamics, an odour particle emitter on the right.
The entity, initially a small green circle with two antennae, was initialised on the left.
The entity had various settings: probe angle, response angle, movement speed, distance, etc.
It would move at the response angle, based on the movement speed and distance, when the probe collided with an odour particle.
This was evolved by way of a simple genetic algorithm, initially asexual reproduction and then sexual reproduction.
This worked really well, it was almost eery to see this blind thing move with apparent intelligence.
