# Movement in the Morley Rules: 
## A (Demonstration) Entry to the [Carle's Game](https://github.com/rivesunder/carles_game) challenge.

<div align="center">
<img src="assets/morley_puffer_r.gif">
<br>
<em>A pattern producing a puffer in Morley (B368/S245) discovered with CMA-ES search</em>
</div>

Greetings apertonauts! This repository describes an evolutionary search for mobile patterns in the Move/Morley Life-like cellular automata rules (B368/S245). I used a covariance matrix adaptation evolution strategy (CMA-ES [Hansen 2016](https://hal.inria.fr/hal-01297037/file/tutorial.pdf)) to find a minimal glider known as "the jellyfish" and also the Morley common puffer. If you want to skip directly to the demonstration of the patterns, use the link below to spin up an interactive bokeh app on [mybinder.org](https://mybinder.org), or clone this repository and open up the notebook at `notebooks/evaluation.ipynb`. For more information on the strategy I used and a discussion of the results, read on. 

[`https://mybinder.org/v2/gh/riveSunder/moving_in_morley/master?urlpath=/proxy/5006/bokeh-app`](https://mybinder.org/v2/gh/riveSunder/moving_in_morley/master?urlpath=/proxy/5006/bokeh-app)

```
# clone this repository and install dependencies, in order to evaluate the found patterns in a Jupyter notebook

# set up a virtual environment with your choice of manager, e.g. virtualenv
virtualenv morley --python=python3
source morley/bin/activate

# clone repo
git clone https://github.com/rivesunder/moving_in_morley
cd moving_in_morley
pip install -e .

# install CARLE
git clone https://github.com/rivesunder/carle
cd carle
pip install -e .
cd ../

# launching jupyter notebook and open notebooks/evaluation.ipynb 
jupyter notebook
```

## A Note on the Rule-String Convention

Life-like cellular automata are defined in grid universes divided into cells, where each cell has a state of either 1 or 0, _aka_ on or off, _aka_ alive or dead. The rules defining the dynamics of a CA determine what conditions lead to a transition from 0 to 1 ("birth") or staying in a state of 1 ("survive"), and all other cells transition to a state of 0 at each time step. Part of what makes a CA Life-like (as in, similar to John Conway's Game of Life) is that the next state of a given cell is fully determined by its current state and the sum of the states of its immediate neighbers, _i.e._ the contents of its Moore neighborhood.

<div align="center">
<img src="assets/moore_neighborhood.png">
<br>
<em>A Moore neighborhood</em>
</div>

We usually write down the rules governing Life-like CA as a rule-string in the format B3/S23. That rule-string, which happens to define Conway's Life, signifies that cells transition from 0 to 1 if they have exactly 3 neighbors, stay 1 if they have 2 or 3 neighbors, and become 0 if they have any other number of neighbors. The Morley rules used in this experiment are written as B368/S245, therefore births (0 to 1) occur for 3, 6, or 8 neighbors, survival (1 to 1) occurs for 2, 4, or 5 neighbors, and all other neighbor counts cause a cell to become 0 if it wasn't already. 

## The Morley/Move Rules B368/S245

The Life-like Morley rule set B368/S245, also known as [Move](https://www.conwaylife.com/wiki/OCA:Move), has many similarities to the seminal totalistic rule set known as [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), B3/S23. Unlike Life, random initializations in a grid subject to the Morley rules tend to quickly settle on still-life and oscillator patterns. For example: 

<div align="center">
<img src="assets/morley_random.gif">
</div>

B368/S245 is capable of producing mobile patterns like gliders and spaceships and is considered Turing complete, as it has been proven to be capable of simulating elementary CA rule 110 by [lazyboi](https://www.conwaylife.com/forums/viewtopic.php?t=&p=102550#p102550) on [conwaylife.com](https://www.conwaylife.com/) forums in August 2020. Several of these patterns are relatively common, including the so-called jellyfish glider shown below:

<div align="center">
<img src="assets/morley_glider_demo.gif">
</div>

# Covariance Matrix Optimization Evolution Strategy

CMA-ES ([Hansen 2016](https://hal.inria.fr/hal-01297037/file/tutorial.pdf)) is an effective optimization strategy applicable to a wide range of optimization tasks, including direct optimization problems like this one in which the aim is to find starting patterns that evolve into mobile machines in the Morley CA. It's also an effective strategy for indirect optimization, _e.g._ optimizing the parameters of a policy that in turn creates patterns (or agent actions in general). In this experiment, CMA-ES was effective at discovering mobile patterns, although it does tend to converge on the same common jellyfish and puffer patterns.  

We can describe the covariance calculation as 

<div align="center">
<img src="assets/covariance_eq.png" width=75%>
</div>


Calculating the covariance matrix in practice is, as is so often the case in modern machine learning, a matter of matrix multiplies:

```
covariance = np.matmul((elite_means - prev_generation_mean).T, (elite_means - prev_generation_mean))
```

For details about my implementation of CMA-ES used in this experiment, have a look at the [source code](https://github.com/riveSunder/carles_game/blob/master/game_of_carle/algos/cma.py), for a succinct explanation check out out Harmaru's [blog post](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/), another implementation in [bevodevo](https://github.com/riveSunder/bevodevo/blob/main/bevodevo/algos/cmaes.py), or Nikolaus Hansen's [tutorial paper](https://hal.inria.fr/hal-01297037/file/tutorial.pdf) for a more detailed view. 

The part of CMA-ES that consumes the most time during training (at least in my hands) is sampling from the resulting multivariate distribution, and not calculating covariance in the first place as I naively expected when I was first learnign about the algorithm. Indeed, distribution sampling can be quite time-consuming, and it limits the utility of CMA-ES for optimizing large numbers of parameters (David Ha gives a heuristic of 1000 parameters being the practical boundary in his [blog on the subject](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/)). CARLE has a default action space of 64 by 64 cell toggles, which comes out to 4096 parameters when optimized directly, and does make for a slow learning process. For faster training, I programmed the `Toggle` agent to only operate within the central 32 by 32 section of the action space, or 1024 parameters, and this turns out to be a good match for my patience and compute resources. As an added benefit, it leaves a buffer zone in between the central pattern that `Toggle` acts on and the boundary of the action space where the reward wrapper begins to consider cell states. This helps to avoid a chaotic boundary bonus, where cells transiently becoming active near the action space boundary generate a reward from the `SpeedDetector` reward wrapper without actually generating interesting mobile patterns. I often see emergence of a chaotic boundary strategy when training policies that currently do have access to the entire action space, like [`CARLA`](https://github.com/riveSunder/moving_in_morley/blob/master/game_of_carle/agents/carla.py) and [`HARLI`](https://github.com/riveSunder/moving_in_morley/blob/master/game_of_carle/agents/harli.py). 

Although a Morley CA universe has a lower chance of producing mobile patterns from random initializations, it does produce an interesting puffer (spaceship that leaves a trail) with somewhat high frequency. This puffer has a period of 170 time steps and leaves behind a trail of simple oscillators. It's also the subject of the [Carle's Game t-shirt](https://rivesunder.threadless.com/designs/puffer-progression), which I plan to send to the first 10 participants in the contest (provided they send me an address that Threadless ships to) not including yours truly, who already has one. 

<div align="center">
<img src="assets/morley_puffer.gif">
</div>

If you want to have a look at the patterns discovered in my direct-optimization CMA-ES experiments, check out the interactive demonstration on mybinder:


[`https://mybinder.org/v2/gh/riveSunder/moving_in_morley/master?urlpath=/proxy/5006/bokeh-app`](https://mybinder.org/v2/gh/riveSunder/moving_in_morley/master?urlpath=/proxy/5006/bokeh-app)

Likewise, if you want to run a similar search experiment, try entering some variation of the command below at the command line, after installing `carles_game` and `carle` and from the `carles_game` root folder:

```
python -m game_of_carle.experiment -mg  32  -ms  1024  -dim  128  -p  16  -v  1  -e  2  -d  cuda:0  -s  13  1337  42   -a  Toggle -w  RND2D  SpeedDetector  -tr  B368/S245  -vr  B368/S245  -tag  glider_search
```

`python -m game_of_carle.experiment` is the name of the experiment module, you can peruse and modify the code [here](game_of_carle/experiment.py). Other arguments are:

* `-mg` or `--max_generations` is the number of generations to train, analagous to an "epoch" in supervised learning.
* `-ms` or `--max_steps` is the number of steps per agent interaction with the environment. You can think of this as the episode length from episodic RL, but CARLE never returns a done signal. 
* `-dim` or `--env_dimension` is the height and width used by the CARLE grid universe. Using a smaller `env_dimension` can make for faster run times, minimum 64. `
* `-p` or `--population_size`, the number of individual agents in the population at each generation.
* `-v` or `--vectorizaiton`, the degree of vectorization to be used in CARLE. This adjust the `N` channel of the `Nx1xHxW` CARLE grid universe and is used to sample `N` grid interactions simultaneously. 
* `-e` or `--episodes`. Each agent will interact with CARLE this many times per generation (for `max_steps` steps)
* `-d` or `--device` is the hardware device to run on, can be `cpu` or `cuda:i` for cuda-enabled set ups, where `i` is the gpu index to use. (No `DataParallel` multi-gpu training at the moment)
* `-s` or `--seeds`. Random seeds used before each experimental run. 3 seeds means the experiment will have 3 seed replicates.
* `-a` or `--agents` are the agent architectures to use in experiments, options are `Toggle`, `HARLI`, or `CARLA` and one or more can be specified.
* `-w` or `--wrappers`. Reward wrappers to be used during training. One or more can be applied, and they are all applied alike for every experimental run in a given experiment. Options are `SpeedDetector`, which gives a reward for changing center of mass of all live cells; `PufferDetector`, rewards growth of the total number of live cells; `RND2D`, which yields a random network distillation exploration bonus (Burda _et al._ 2018](https://arxiv.org/abs/1810.12894v1)); or `AE2D`, which gives an exploration bonus based on autoencoder loss. Note that `AE2D` is a translation invariant reward and `RND2D` is not, due to the use of fully connected layers in the random and prediction networks.
* `-tr` or `--training_rules` are the rules, specified in CA rulestring format, to be used during training. If more than one set of rules are supplied (_e.g._ `-tr B3/S23 B3/S023`), each experimental run will sample from the available options each time the environment is fully reset.
* `-vr` or `--validation_rules` are the validation rules. More than one can be specified and every validation rule set will be examined during validation (currently every 16th generation). 
* `-tag` is a string tag to make it easier to search for your experiment on tensorboard. 

## Appendum

<em>
Note that this is not a real entry to the [Carle's Game](https://github.com/rivesunder/carles_game) competition, as the progenitor of the contest and myself are one and the same. It is, however, an example of how one might go about addressing the challenge and writing up an experiment in order to enter the contest. My intention with this example is that it will pique your interest and encourage the reader to participate in the contest as an entrant or judge, and serve as a reminder that you don't have to solve open-endedness or AI in order to make a good entry to the challenge. A relatively simple and inconclusive exploration like the one described here is a welcome contribution. In fact, you're welcome and encouraged to take advantage of any of the tools provided in this repo or its parent in order to explore the world of machine interaction (or machine-aided interaction) with Life-like cellular automata in CARLE. 
</em>

<em>
Good luck, thanks for reading, and I look forward to seeing what your creations create.  
</em>

