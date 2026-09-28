# Performance-Aware Safety Filters
This repository presents the implementation of performance-aware predictive safety filters

## Parameters

### UAV Physical Parameters

For the UAV physical parameters, we use $m=1~\mathrm{kg}$, $g=9.81~\mathrm{m/s^2}$, $I_x=I_y=0.02~\mathrm{kg\,m^2}$, $I_z=0.04~\mathrm{kg\,m^2}$, and $d_x=d_y=d_z=0.2~\mathrm{kg/s}$.

### LQR Controller

The LQR controller uses $Q = \mathrm{diag}(2,2,2,50,50,50,5,5,5)$ and $R = \mathrm{diag}(0.05,5,5,2)$, and is applied with a sampling time of $0.01~\mathrm{s}$.

### DDPG Network and Training Setup

Both the actor and critic networks have two hidden layers of 32 neurons each, using GeLU activations except for the actor's output layer, which uses $\tanh$.

Training uses a discount factor of $0.99$, actor and critic learning rates of $0.001$ and $0.002$, respectively, a target network update rate of $0.005$, a batch size of $64$, a replay buffer of size $10^5$, $2000$ training episodes, a maximum episode length of $300$ steps, and is applied with a sampling time of $0.1~\mathrm{s}$.

### Predictive Safety Filter

The predictive safety filter uses a prediction horizon of $10$ steps, $\lambda_1=10$, $\lambda_2=1$, subject to the state constraints $-5 \leq p_x \leq 5$, $-5 \leq p_y \leq 5$, $0 \leq p_z \leq 10$, the input constraints $-0.5 \leq v_{\mathrm{com,x}} \leq 0.5$, $-0.5 \leq v_{\mathrm{com,y}} \leq 0.5$, $-0.1 \leq v_{\mathrm{com,z}} \leq 0.1$, and the rate constraints $-0.1 \leq \Delta v_{\mathrm{com,x}} \leq 0.1$, $-0.1 \leq \Delta v_{\mathrm{com,y}} \leq 0.1$, $-0.1 \leq \Delta v_{\mathrm{com,z}} \leq 0.1$.

In both filters, we constrain the safety filter computation to complete within the sampling time of the DDPG agent.

The slack variable penalization weight in both the standard and performance-aware safety filters is manually tuned to optimize performance, resulting in a weight of $4$ for the standard safety filter and $440$ for the performance-aware safety filter.

## Repository structure

- `DRN_1_Env.py`: Drone environment (Gymnasium), including the LQR inner-loop controller
- `DRN_2_DDPG.py`: DDPG agent and training
- `DRN_3_Plot.py`: Plotting utilities
- `DRN_4_PASF.py`: Performance-aware safety filter (PASF)
- `DRN_4_SSF.py`: Standard safety filter (SSF)
- `DRN_5_Sims.py`: Runs simulations for all scenarios (initial conditions and obstacle radii)
- `DRN_6_Data.py`: Generates the bar plot of episodic returns versus obstacle radius
- `DRN_7_Draw.py`: Generates XY-plane trajectory plots for different initial conditions


## Usage

Install the dependencies:

```bash
pip install do-mpc casadi scipy matplotlib numpy torch gymnasium
```

All scripts are in `src/` and use relative paths, so run them from there.

Pretrained weights and the results used in the paper are included, so each step below can be run on its own.

1. **Train the DDPG agent (optional).** Run `python DRN_2_DDPG.py`. This saves `ddpg_quad_full_state.pth` and `nn_weights_quad_2_effort_300.npz`. The safety filters load `nn_weights_quad_2_effort_prcp_300.npz`, so rename the new weights file to that name if you want the filters to use it.

2. **Run a single simulation.** Run `python DRN_4_PSF.py` for the performance-aware filter or `python DRN_4_SSF.py` for the standard filter. Optional arguments:

   | Argument | Description | Default |
   |---|---|---|
   | `-ocx`, `-ocy` | Obstacle center | `-2.0`, `-2.0` |
   | `-or` | Obstacle radius | `0.5` |
   | `-xi`, `-yi`, `-zi` | Initial position | near `(-3.6, -3.8, 3.0)` |
   | `-rid` | Run identifier | none |

   Each run appends its episode return and solver timings to `psf_quad_results.jsonl` and saves the trajectory to `psf_quad_trajectory_PASF_<run_id>.npz`.

3. **Run all scenarios.** Set `SCRIPT` in `DRN_5_Sims.py` to `DRN_4_PSF.py` or `DRN_4_SSF.py`, then run `python DRN_5_Sims.py`. This runs 30 simulations: obstacle radii 0.5, 0.75 and 1.0, each with 10 perturbed initial positions. Run IDs 0–9, 10–19 and 20–29 correspond to the three radii in that order. Both filters write to the same output files, so rename the results after each batch:
   - Performance-aware: `psf_quad_results_GenPASF_org.jsonl`
   - Standard: `psf_quad_results_baseline_org.jsonl`

4. **Plot episodic returns versus obstacle radius.** Run `python DRN_6_Data.py`. This reads the two `.jsonl` files above and saves `psf_quad_mean_std_grouped_compare.png`.

5. **Plot XY trajectories.** Run `python DRN_7_Draw.py`. This reads `Sam_1_0.npz` to `Sam_1_9.npz` (standard filter) and `Sam_3_0.npz` to `Sam_3_9.npz` (performance-aware filter) and saves `psf_quad_groups.png`. To plot your own runs, rename the trajectory files of one obstacle radius to these names, or point the script to them with `--prefix1` and `--prefix2`.