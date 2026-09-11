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

The slack variable penalization weight in both the standard and performance-aware safety filters is manually tuned to optimize performance, resulting in a weight of $4$ for the standard safety filter and $400$ for the performance-aware safety filter.