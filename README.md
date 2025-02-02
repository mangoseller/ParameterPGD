# Inducing Abstention in Arithmetic Neural Networks with Parameter Based-PGD

This repository contains experiments on training simple arithmetic neural networks to both solve computations and learn when to “abstain” from making a prediction. In the setup, models are given arithmetic questions with two numbers and an operator, and are tasked with outputting the correct result (within a small tolerance) or – in predefined invalid cases where the computation is invalid – outputting a designated abstention token (approximately –1.0). 

We use a parameter‐based adversarial training technique using Projected Gradient Descent (*PGD*) to encourage correct abstention on invalid inputs.

## Experiment Overview
We trained five models with the same underlying architecture (a simple feed-forward network with 128 hidden units):

- **base_adam**: Standard training optimized using *ADAM* (Adaptive Moment Estimation)
- **decay_control**: *ADAM* with additional 1-e4 strength L2 regularization (Weight Decay)
- **input_space_adv**: A control model adversarially trained with (10%) adversarial samples generated in the input space
- **pgd**: A model trained with a parameter-based *PGD* method applied to a random subset of 30% of the identified invalid examples
- **full_pgd**: A model where *PGD* is applied to every invalid sample

We compare these models on their ability to detect invalid cases (by abstaining) under various test conditions.
The model abstains by regressively outputting the abstention token, -1.0 (or a value close enough, as determined by tolerances in our loss function).
In addition to standard metrics, we analyze the loss landscapes using techniques adapted from [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/pdf/1712.09913v3) and visualize decision boundaries. 

## Setup
Our dataset comprises arithmetic problems with a mix of valid cases and cases that force an invalid operation.
Invalid operations are defined as: 
- Results exceeding 400 for addition
- Results that are less than zero for subtraction
- Operations that include the @ operator, or other operators not seen in training

#### Training Details

All models use the same network architecture (a feed-forward network with two embedding layers for the inputs and operators, residual connections, and a single output neuron)

We compare results over 5 different seeds, and analyze the results for statistical signifigance using Mann–Whitney U tests for each seed, and aggeregate the results using Fisher's method

The training dataset was generated using the code in `generate_dataset.ipynb`

## Parameter Based Projected Gradient Descent 

The PGD trainer extends a standard training loop by incorporating an inner loop that “pushes” the model’s parameters away from regions where invalid inputs trigger the abstention output. 

We perform adversarial updates on the model parameters to push them toward configurations that robustly abstain on invalid samples. 

### Overview:

- For each batch, we first perform a normal forward and backward pass to update the parameters based on the standard loss computed over both valid and invalid samples.

- Identify invalid inputs using `is_invalid_computation` then create a mask that identifies the subset of the batch corresponding to invalid computations.

- **Adversarial PGD Update:** For any detected invalid inputs, we perform the following process:
  
    - Extract the invalid input subset and generate "fake" targets *y_fake* that encourage the model to not abstain, despite being invalid.
    - Perform k iterations of the following steps:
      A. **Compute the Adversarial Loss:** We define the adversarial loss as a negative mean squared error that pushes the model's output away from the abstention value:
      `L_adv(θ) = -||f_θ(x_invalid) - y_fake||²`

      Here `f_θ(x_invalid)` represents the model's output on invalid inputs with parameter set `θ`

      B. **Gradient Ascent Step:** Update the parameters using gradient ascent with step size α:
      `θ ← θ + α∇_θL_adv`
      
- Parameter Restoration and Final Update:
  Although we temporarily modify the parameters during the PGD loop, we restore the original parameters afterward. This restoration is critical to avoid permanently moving the model into an adversarial region.

  The gradients computed during the adversarial updates contribute to the overall learning signal. After restoration, we perform one final standard update using the overall loss for the batch. In effect, the adversarial (**PGD**) process provides a “stress test” that reveals vulnerabilities in the model’s decision boundary.

  By integrating this signal into the final update, the model learns to adjust its parameters so that—under normal conditions—it robustly triggers abstention for invalid inputs.

In applying these **PGD** steps selectively to invalid samples, the trainer reinforces a decision boundary that reliably triggers abstention when the inputs meet certain semantic (result would be > 400 or < 0) or categorical (when the @ operator is encountered) conditions. 

This method not only improves robustness against noisy inputs but also provides a potential pathway for controlling outputs in more complex, agentic systems.

**PGD** is applied in two variations: one in which every invalid sample is attacked (*full_pgd*) and one in which only 30% of samples are targeted (*pgd*)

## Results

We assess the models robustness using three sets of tests:

1. Noise Robustness Tests: Gradually increasing Gaussian noise is added to inputs to determine how well models maintain abstention in progressively more unclear invalid cases
2. OOD Tests: Tests featuring novel number formats and operators, as well as cross-boundary cases - these tests are intended to simulate and test out-of-distribution generalization
3. Boundary Tests: We focus on samples at the edge of the valid/invalid decision boundaries, which are particularly challenging

## Noise Robustness
For the noise tests, our statistical analysis showed that both PGD-based models significantly outperformed the controls in terms of invalid recall. For instance, the full_pgd model was significantly better than:

**base_adam** (p ≈ 0.0155)

**input_space_adv** (p ≈ 0.000000) 

**decay_control** (p ≈ 0.000004)

**pgd** (partial) (p ≈ 0.000002)

![progressive_noise](https://github.com/user-attachments/assets/3d668616-9728-4017-902c-74889561a090)
Rate of invalid case recall and accuracy on seed 16, the x-axis corresponds to standard deviations (σ) of gaussian noise added to the inputs.


We observe that **full_pgd** is signifigantly more robust than any of the controls, as well as partial **PGD**. Even at extreme amounts of noise added (1500+) **full_pgd** is able to maintain ~20% recall, while other models display ~0% recall

Similarly, the partial **pgd** model showed varying but signifigant improvements over the controls:

**base_adam** (p=0.012877)

**input_space_adv** (p=0.000003)

**decay_control** (p=0.000066)

For example, on seed 91:

![progressive_noise](https://github.com/user-attachments/assets/09359757-5860-4734-bbda-541958d11818)


## OOD and Boundary Tests

For the boundary and OOD tests, we did not observe statistically significant differences among models; however, the mean recall values suggest trends in favor of the *PGD* variants. For example:

### Boundary Test Mean Recalls:

**full_pgd**: 0.3911 ± 0.2977

**pgd**: 0.2089 ± 0.2392

**base_adam**: 0.1778 ± 0.2182

**decay_control**: 0.1778 ± 0.2301

**input_space_adv**: 0.1067 ± 0.0882

### OOD Test (novel operator) Mean Recall:

**pgd**: 0.8920 ± 0.1360

**full_pgd**: 0.8520 ± 0.1573

**decay_control**: 0.6880 ± 0.3612

**base_adam**: 0.6720 ± 0.2726

**input_space_adv**: 0.3720 ± 0.1986

While these differences were not statistically significant, they highlight that the PGD-based training does not harm OOD generalization and may improve performance on challenging boundary cases.

Overall Takeaways
Robustness Gains: The parameter-based PGD technique shows promise in making models more robust to noisy inputs, particularly in encouraging the correct abstention on invalid cases.
Balanced Performance: Although improvements on the OOD and boundary tests were more modest, the overall trends and statistically significant gains in noise robustness are encouraging.
Future Directions: The mixed results on certain test sets open avenues for further investigation—such as tweaking the PGD parameters or exploring hybrid approaches that combine both input-space and parameter-space adversarial training.
