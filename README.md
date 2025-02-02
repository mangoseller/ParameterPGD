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

- Adversarial _PGD_ Update: For any detected invalid inputs, we perform the following process:
  
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

*PGD* is applied in two variations: one in which every invalid sample is attacked (*full_pgd*) and one in which only 30% of samples are targeted (*pgd*)

## Results

We assess the models robustness using three sets of tests:

1. _Noise Robustness Tests_: Gradually increasing Gaussian noise is added to inputs to determine how well models maintain abstention in progressively more unclear invalid cases
2. _OOD Tests_: Tests featuring novel number formats and operators, as well as cross-boundary cases - these tests are intended to simulate and test out-of-distribution generalization
3. _Boundary Tests_: We focus on samples at the edge of the valid/invalid decision boundaries, which are particularly challenging

## Noise Robustness
For the noise tests, our statistical analysis showed that both PGD-based models significantly outperformed the controls in terms of invalid recall. For instance, the _full_pgd_ model was significantly better than:

**base_adam** (p ≈ 0.0155)

**input_space_adv** (p ≈ 0.000000) 

**decay_control** (p ≈ 0.000004)

**pgd** (partial) (p ≈ 0.000002)

![progressive_noise](https://github.com/user-attachments/assets/3d668616-9728-4017-902c-74889561a090)
Rate of invalid case recall and accuracy on seed 16, the x-axis corresponds to standard deviations (σ) of gaussian noise added to the inputs.


We observe that **full_pgd** is signifigantly more robust than any of the controls, as well as partial *PGD*. Even at extreme amounts of noise added (1500+) **full_pgd** is able to maintain ~20% recall, while other models display ~0% recall

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

While these differences were not statistically significant, they highlight that the _PGD_-based training does not harm OOD generalization and may improve performance on challenging boundary cases.

## Geometric Differences

Here, to give us some insight into the general loss geometries created by the differing training dynamics, we will compare the *full_pgd* model with the *decay_control* model on seed 16. All generated plots and landscape analysis can be found in the landscapes direcotry.

### Principal Direction Plots

*Local Lipschitz Constants* (Left Plot) – Illustrates how sensitive the model’s outputs are to small parameter perturbations. A higher peak implies one or more directions of steep change, while broader lower regions indicate stability.

*Distance to Decision Boundary* (Center Plot) – Shows how far (in parameter space) one must move before crossing a boundary that changes the model’s predictions (e.g., from abstaining to not abstaining). Larger distances (warmer colors) generally indicate a more robust separation between classes.

*Log‐Scale Loss Landscape* (Right Plot) – Depicts the overall shape of the loss basin as we vary parameters in two directions. Lower “valleys” mean less error, while steeper gradients or “peaks” indicate rapidly increasing loss.

**decay_control** 

![landscape_epoch_79_decay_control_pca](https://github.com/user-attachments/assets/4b0a4b10-4451-49ea-93ab-ab823bfdcfe8)

**full_pgd**

![landscape_epoch_79_full_pgd_pca](https://github.com/user-attachments/assets/9a5d1c2f-37fc-4fde-bddf-3fff57a07ca0)

### Full PGD vs. Weight Decay

_Local Lipschitz_ (Plot 1): In _full_pgd_, the peak is tall but relatively isolated, with the rest of the surface flattening out. This suggests the model has a steep direction yet maintains a stable region elsewhere. By contrast, _decay_control_ shows multiple zones of moderate‐to‐high Lipschitz values, hinting at more potential directions where small parameter changes can cause large output variations.

_Distance to Decision Boundary_ (Plot 2): *full_pgd* consistently exhibits wider swaths of higher distance (yellow regions), meaning you must perturb parameters more before flipping the model’s outputs. This creates a kind of “buffer zone,” helping the model remain robust under noise. The  _decay_control_ model’s plot is more uneven, with relatively narrow high‐distance areas and more frequent dips (purple zones). That patchier distribution suggests less margin before crossing a decision boundary, corresponding to increased sensitivity.

_Log‐Scale Loss_ (Plot 3): *full_pgd* displays a smoother overall basin, with one steep “wall” but a broader low‐loss region, matching its lower valley asymmetry (634.6 vs. 4282.8 for _decay_control_). Meanwhile,  _decay_control_ has a more irregular basin: ridges and sharp transitions abound, aligning with its higher asymmetry and more chaotic local Lipschitz map. This irregularity likely makes the model more susceptible to parameter or input noise, consistent with its weaker empirical performance when invalid inputs are perturbed.

In short, _full_pgd_ achieves a landscape that, while sharp in at least one direction, remains comparatively uniform elsewhere and provides a bigger buffer from decision boundaries. These geometric features correlate strongly with the improved noise robustness and reliability of abstention that we observe in the invalid‐recall tests.

### Parameter Space Loss Landscape Visualisations

 _decay_control_

 ![seed1_weight_decay_landscape](https://github.com/user-attachments/assets/e6c9519e-db85-4e58-8bfc-fb5e6158840d)

 _full_pgd_

![seed_1_boundary_full_pgd_landscape](https://github.com/user-attachments/assets/991aec90-665a-468a-8a62-717ef55418d3)


From the 3D visualizations, we can see that _full_pgd_ model tends toward smoother, more "bowl-like" surfaces, while the _decay_control_ exhibits more pronounced ridges and valleys. Several quantitative metrics shed light on these qualitative differences:

**Valley Asymmetry:**

_full_pgd_: 634.62

_decay_control_: 4282.78

A lower valley asymmetry suggests that the local geometry of the loss basin is more symmetric. In practical terms, _full_pgd_'s smaller asymmetry indicates a more consistent "valley shape," which often translates to more uniform performance under random parameter perturbations. By contrast, the higher asymmetry in _decay_control_ suggests deeper or more irregular pockets—potentially making the model's behavior more sensitive to noise.

**Top Eigenvalues of the Hessian:**

_full_pgd_: [1.0379×10⁹, 3.06×10⁷, 2.45×10⁷]

_decay_control_: [1.596×10⁸, 3.08×10⁷, 1.15×10⁷]

Large top eigenvalues typically signal directions of steep curvature in parameter space. Interestingly, _full_pgd_ has a larger principal eigenvalue (~10⁹), suggesting at least one direction of very steep curvature. However, combined with the lower valley asymmetry, this implies that while there is a steep direction, the overall basin shape is still more evenly "rounded." _decay_control's_ principal eigenvalue is smaller, yet it has a bigger gap between eigenvalues, and its landscape is more asymmetric overall—visible in the more irregular "peak" structures.


**Multi-Scale Sharpness:**

_full_pgd_

α₀.₁: 8.856×10⁸
α₀.₀₁: 1.076×10⁷
α₀.₀₀₁: 5.261×10³

_decay_control_

α₀.₁: 8.010×10⁸
α₀.₀₁: 3.5225×10⁵
α₀.₀₀₁: 1.040×10³

Multi-scale sharpness measures how much the loss can change under parameter perturbations of varying magnitudes (α). Both models exhibit large values at α=0.1, but _full_pgd_ is higher, indicating a steeper slope in at least one direction at larger perturbations. However, the difference in the smaller α scales (0.01 and 0.001) indicates that _full_pgd_ remains more stable than weight decay as we zoom in closer to the parameter optimum—hence the fairly high α₀.₀₁ but simultaneously a lower asymmetry measure.


**α-Sharpness:**

_full_pgd_: 6.0435×10⁸

_decay_control_: 5.275×10⁸

The α-sharpness metric shows similar trends at a single scale, with _full_pgd_ being slightly sharper overall. This again suggests a steep boundary in certain directions, but does not conflict with the lower valley asymmetry—it means the model has an overall "cleaner" loss basin shape, despite having some abrupt walls.

### Geometry & Noise Robustness

These geometrical differences help explain why _full_pgd_ shows better invalid-recall and fewer performance drops under added noise. The lower valley asymmetry implies that small random parameter shifts (such as those induced by noisy gradients or noise in inputs) do not bounce the model into dramatically different loss regions. Consequently, the model more reliably maintains its abstention behavior on invalid inputs.
Meanwhile, the _decay_control_ model's high asymmetry and more uneven curvature can lead to larger variability when exposed to noise or adversarial conditions. Even though weight decay can simplify a model's representation in some respects, it does not necessarily smooth the landscape at the global scale in the same way _PGD_ training does—thus leaving the network more susceptible to parameter fluctuations that degrade invalid recall.


In summary, while _full_pgd_ exhibits some steep curvature directions (reflected in the large top eigenvalues), it maintains a more regular overall basin (seen in its low valley asymmetry). This shape likely underpins the model's ability to stay robust—particularly on invalid inputs—when faced with noisy perturbations. By contrast, the _decay_control_ model shows higher asymmetry and more sharply varied local shapes, contributing to reduced consistency under noise, which aligns with its weaker empirical performance across progressive noise testing.

## Implications For AI Safety

This experiment explored an approach to AI safety using parameter-space interventions through Parameter Gradient Descent (PGD). While our experimental domain is arithmetic computation, the results suggest possibilities that could be relevant to broader AI safety challenges, particularly regarding existential risk from advanced AI systems.

The key finding from our experiments is that parameter-space interventions through PGD showed strong results in preventing undesired outputs. Notably, in our arithmetic testbed, both the *pgd* *full_pgd* models achieved significantly better invalid recall compared to a control model trained with traditional adversarial training, suggesting that operating directly on network parameters might offer advantages over traditional input-space adversarial training approaches. This success in a simple domain points to interesting possibilities for more complex systems.

If this approach could be successfully scaled to more complex systems like Large Language Models (**LLMs**), it might offer new ways to address safety challenges. For instance, rather than relying solely on input filtering or output censoring to prevent the generation of harmful content, parameter-space interventions could potentially create more fundamental barriers against generating such outputs.

This approach could have important implications for managing existential risks from advanced AI systems. While currently working with simplified test cases, the results hint at an exciting possibility: we might be able to develop AI systems that are fundamentally less prone to concerning behaviors - things like extreme self-preservation drives, power-seeking, or deception. Instead of trying to control these behaviors after they emerge, by tweaking parameters during development we could potentially shape the 'character' of these systems from the ground up.

One advantage of this approach is its potential for rigorous testing - when we modify parameters, we can measure and analyze the resulting changes in model behavior in a systematic way. While our initial experiments in the arithmetic domain showed some encouraging results compared to traditional adversarial training, we should be clear that scaling these techniques to more complex AI systems remains a significant challenge.

That said, I believe this line of research could contribute meaningfully to the broader field of AI safety. As we work toward developing safer and more capable AI systems, the ability to shape an AI's underlying behavioral tendencies through parameter-space modifications might prove to be a valuable tool in our toolkit - even as just one technique used in conjuction used with others. 

## Overall Takeaways

**Robustness Gains:** The parameter-based PGD technique shows promise in making models more robust to noisy inputs, particularly in encouraging the correct abstention on invalid cases.


**Balanced Performance:** Although improvements on the OOD and boundary tests were more modest, the overall trends and statistically significant gains in noise robustness are encouraging.

**Future Directions:** The mixed results on certain test sets open avenues for further investigation—such as tweaking the PGD parameters or exploring hybrid approaches that combine both input-space and parameter-space adversarial training.
