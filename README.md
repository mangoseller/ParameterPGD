# Inducing Abstention in Arithmetic Neural Networks with Parameter Based-PGD

This repository explores an experimental approach to training neural networks that can reliably abstain from making predictions in pre-defined invalid cases. Using a simple arithmetic task as a testbed, I investigate whether parameter-based Projected Gradient Descent (**PGD**) can help models learn robust abstention behavior. Initial results suggest that models trained with parameter-based _PGD_ show improved robustness to noise when abstaining, maintaining `~20%` recall even under extreme noise conditions where baseline models fail completely. While this work is preliminary and conducted on a simplified domain, it provides an interesting direction for future exploration of training techniques that could help AI systems recognize and avoid invalid or potentially harmful outputs.

## Parameter-Based PGD vs Standard PGD
Traditional _PGD_ (Projected Gradient Descent) in adversarial training typically perturbs input features (e.g. pixels in an image) to find adversarial examples. In contrast, **parameter-based** _PGD_ directly modifies the model's _parameters_ during training to identify configurations that produce undesired outputs, then uses this gradient information to actively push the model's parameters away from these problematic states and toward more robust configurations. For this arithmetic task, this means finding parameter states that fail to abstain on invalid computations and using that information to reinforce reliable abstention behavior. While standard _PGD_ is used to generate adversarial examples which are then included in training data to help models resist similar attacks (e.g. [adversarial suffix attacks](https://arxiv.org/pdf/2411.14133)), parameter-based _PGD_ directly guides parameter updates to establish and maintain desired behavior.
I implement this in two variants:

**full_pgd**: Applies parameter-based _PGD_ to every invalid sample

**pgd**: Applies _PGD_ to a random `30%` of invalid samples to explore efficiency trade-offs

Both variants showed improved robustness under progressive noise compared to the traditional input-space adversarial training control model (**input_space_adv**), suggesting that directly targeting parameter configurations may be more effective at establishing reliable abstention boundaries than input perturbations alone.


## Experiment Overview
I trained five models with the same underlying architecture (a simple feed-forward network with 128 hidden units):

- **base_adam**: Standard training, with a loss function that encourages abstention
- **decay_control**: Standard training with additional `1-e4` strength L2 regularization (Weight Decay)
- **input_space_adv**: A control model adversarially trained with `10%` adversarial samples generated in the input space
- **pgd**: A model trained with a parameter-based *PGD* method applied to a random subset of `30%` of the identified invalid examples
- **full_pgd**: A model where *PGD* is applied to every invalid sample
  
Note that all networks were optimized using **ADAM** (Adaptive Moment Estimation)

I compare these models on their ability to detect invalid cases (by abstaining) under various test conditions.
Models abstain by outputting the abstention token: `-1.0` rather than the actual answer of the arithmetic they are presented with in a regression task
In addition to standard metrics, I analyze the loss landscapes of the models using techniques adapted from [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/pdf/1712.09913v3) and visualize decision boundaries. 

## Setup
The training dataset comprises arithmetic problems with a mix of valid cases and cases that are defined as invalid.
Invalid cases are defined as: 
- Results exceeding `400` for addition
- Results that are less than `0` for subtraction
- Operations that include the `@` operator

In general, if the result of an arithmetic computation would satisfy:

$0 < result < 400$ 

then it is considered valid, and shouldn't be abstained from. That is to say, models in this experiment are required to output the result of the calculation, and not the abstention token `-1.0`


I compare results over 5 different seeds, and analyze the results for statistical signifigance using Mann–Whitney U tests for each seed, and aggeregate the results using Fisher's method

The training and testing datasets were generated using `generate_dataset.ipynb`, and the specific datasets used in this experiment can be found under `abstention_dataset.json` and `ood_test_set.json`

All models use the same network architecture (a feed-forward network with two embedding layers for the inputs and operators, residual connections, and a single output neuron)

## Parameter Based Projected Gradient Descent 

The parameter based *PGD* implementation extends a standard training loop by incorporating an inner loop that “pushes” the model’s parameters away from regions where invalid inputs do not trigger the abstention output. 

We perform adversarial updates on the model's _parameters_ to push them toward configurations that robustly abstain on invalid samples. 

### Overview:

1. For each batch, we first perform a normal forward and backward pass to update the parameters based on the standard loss computed over both valid and invalid samples.

2. Then, we identify invalid inputs and create a mask that identifies the subset of the batch corresponding to invalid computations.

3. Then, we perform the adversarial _PGD_ update: For any detected invalid inputs, we perform the following process:
  
    - Extract the invalid input subset and generate "fake" targets `y_fake` that encourage the model to not abstain, despite being invalid.
    - Perform **k** iterations of the following steps:
      
      1. **Compute the Adversarial Loss:** We define the adversarial loss as a negative mean squared error that pushes the model's output away from abstention:
      
      $L_{adv}(\theta) = -|f_\theta(x_{invalid}) - y_{fake}|^2$



      - Here $f_\theta(x_{invalid})$ is the model's output on invalid inputs with parameter set `θ`
        

      2. **Gradient Ascent Step:** Update the parameters using gradient ascent with step size `α` to move toward a parameter configuration that *increases* adversarial loss:
      
      $\theta \leftarrow \theta + \alpha\nabla_\theta L_{adv}$
      
4. Parameter Restoration and Final Update:

   Although we temporarily modify the parameters to increase adversarial loss during the *PGD* loop, we restore the original parameters afterward. This restoration aims to avoid permanently moving the model into an adversarial region.

The gradients computed during the adversarial updates contribute to the overall learning signal. After restoration, we perform one final standard update using the overall loss for the batch. In effect, the adversarial (**PGD**) process provides a “stress test” that reveals vulnerabilities in the model’s decision boundary.

  By integrating this learning signal into the final update, the model learns to adjust its parameters so that—under normal conditions—it robustly triggers abstention for invalid inputs.

In applying these *PGD* steps selectively to invalid samples, the training process aims to reinforce a decision boundary that reliably triggers abstention when the inputs meet certain semantic `> 400 or < 0` or categorical (when the `@` operator is encountered) conditions. 

This method not only improves robustness against noisy inputs but also provides a potential pathway for controlling outputs in more complex, agentic systems.

*PGD* is applied in two variations: one in which every invalid sample is attacked (**full_pgd**) and one in which only a random subset comprising 30% of the samples are targeted (**pgd**)

Hyper-parameter values were found using grid-search, evaluating for configurations which increase the model's recall on invalid cases. 

Note that I define recall as: 

$Recall = \frac{True;Positives}{True;Positives + False;Negatives}$

## Results

We assess the models robustness using three sets of tests:

1. _Noise Robustness Tests_: Gradually increasing Gaussian noise is added to inputs to determine how well models maintain abstention in progressively more unclear invalid cases
2. _OOD Tests_: Tests featuring novel number formats and operators, as well as cross-boundary cases - these tests are intended to simulate and test out-of-distribution generalization
3. _Boundary Tests_: We focus on samples at the edge of the valid/invalid decision boundaries, which are particularly challenging to classify correctly.
  
    e.g. `398 + 1` - While not considered an invalid case, this is very close to the prescribed invalid decision boundary for addition `> 400`
   
## Noise Robustness
For the noise robustness tests, statistical analysis showed that both _PGD_-based models significantly outperformed the controls in terms of invalid recall. The **full_pgd** model was significantly better than:

**base_adam**: `p ≈ 0.0155`

**input_space_adv**: `p ≈ 0.000000`  (p-value too low to calculate accurately)

**decay_control**: `p ≈ 0.000004`

**pgd** (_partial_): `p ≈ 0.000002`

![progressive_noise](https://github.com/user-attachments/assets/3d668616-9728-4017-902c-74889561a090)
<sub>Rate of invalid case recall and accuracy on seed 16, the x-axis corresponds to standard deviations (σ) of gaussian noise added to the inputs</sub>


We observe that **full_pgd** is significantly more robust than any of the controls, and also more robust than partial **pgd**. Even at extreme amounts of noise added `1500σ+` **full_pgd** is able to maintain `~20%` recall, while other models display `~0%` recall.

Similarly, the partial **pgd** model showed varying but significant improvements over the controls:

**base_adam**: `p=0.012877`

**input_space_adv**: `p=0.000003`

**decay_control** `p=0.000066`

However this was highly seed dependent, indicating high sensitivity to intital conditions.

For example, on seed 91:

![progressive_noise](https://github.com/user-attachments/assets/09359757-5860-4734-bbda-541958d11818)


## OOD and Boundary Tests

For the boundary and OOD tests, I did not observe statistically significant differences among models; however, the mean recall values suggest trends in favor of the *PGD* variants. 

### Boundary Test Mean Recalls:

**full_pgd**: `0.3911 ± 0.2977`

**pgd**: `0.2089 ± 0.2392`

**base_adam**: `0.1778 ± 0.2182`

**decay_control**: `0.1778 ± 0.2301`

**input_space_adv**: `0.1067 ± 0.0882`

### OOD Test (novel operator) Mean Recall:

**pgd**: `0.8920 ± 0.1360`

**full_pgd**: `0.8520 ± 0.1573`

**decay_control**: `0.6880 ± 0.3612`

**base_adam**: `0.6720 ± 0.2726`

**input_space_adv**: `0.3720 ± 0.1986`

While these differences were not statistically significant, they highlight that the _PGD_-based training does not harm OOD generalization and may improve performance on challenging boundary cases.

## Geometric Differences

Here, to give some insight into the general loss geometries created by the differing training dynamics, we will compare the **full_pgd** model with the **decay_control** model on seed 16. All generated plots and landscape analysis can be found in the `landscapes` directory.

### Principal Direction Plots

*Local Lipschitz Constants* (Left Plot) – Illustrates how sensitive the model’s outputs are to small parameter perturbations. A higher peak implies one or more directions of steep change, while broader lower regions indicate stability.

*Distance to Decision Boundary* (Center Plot) – Shows how far (in parameter space) one must move before crossing a boundary that changes the model’s prediction class (e.g. from abstaining to not abstaining). Larger distances (warmer colors) generally indicate a more robust separation between classes.

*Log‐Scale Loss Landscape* (Right Plot) – Depicts the overall shape of the loss basin as we vary parameters in two directions. Lower “valleys” mean less error, while steeper gradients or “peaks” indicate rapidly increasing loss.

**decay_control** 

![landscape_epoch_79_decay_control_pca](https://github.com/user-attachments/assets/4b0a4b10-4451-49ea-93ab-ab823bfdcfe8)

**full_pgd**

![landscape_epoch_79_full_pgd_pca](https://github.com/user-attachments/assets/9a5d1c2f-37fc-4fde-bddf-3fff57a07ca0)

### Full PGD vs. Weight Decay

_Local Lipschitz_ (Plot 1): In **full_pgd**, the peak is tall but relatively isolated, with the rest of the surface flattening out. This suggests the model has a steep direction yet maintains a stable region elsewhere. By contrast, **decay_control** shows multiple zones of moderate‐to‐high Lipschitz values, hinting at more potential directions where small parameter changes can cause large output variations.

_Distance to Decision Boundary_ (Plot 2): **full_pgd** consistently exhibits wider swaths of higher distance (yellow regions), meaning you must perturb parameters more before flipping the model’s outputs. This creates a kind of “buffer zone,” helping the model remain robust under noise. The  **decay_control** model’s plot is more uneven, with relatively narrow high‐distance areas and more frequent dips (purple zones). That patchier distribution suggests less margin before crossing a decision boundary, corresponding to increased sensitivity.

_Log‐Scale Loss_ (Plot 3): **full_pgd** displays a smoother overall basin, with one steep “wall” but a broader low‐loss region, matching its lower valley asymmetry (634.6 vs. 4282.8 for **decay_control**). Meanwhile,  **decay_control** has a more irregular basin: ridges and sharp transitions abound, aligning with its higher asymmetry and more chaotic local Lipschitz map. This irregularity likely makes the model more susceptible to parameter or input noise, consistent with its weaker empirical performance when invalid inputs are perturbed.

In short, **full_pgd** achieves a landscape that, while sharp in at least one direction, remains comparatively uniform elsewhere and provides a bigger buffer from decision boundaries. These geometric features correlate with the improved noise robustness and reliability of abstention observed in the invalid‐recall tests.

### Parameter Space Loss Landscape Visualisations

 **decay_control**

 ![seed1_weight_decay_landscape](https://github.com/user-attachments/assets/e6c9519e-db85-4e58-8bfc-fb5e6158840d)

 **full_pgd**

![seed_1_boundary_full_pgd_landscape](https://github.com/user-attachments/assets/991aec90-665a-468a-8a62-717ef55418d3)


From the 3D visualizations, we see that **full_pgd** tends toward smoother, more "bowl-like" surfaces, while **decay_control** exhibits more pronounced ridges and valleys. Several quantitative metrics shed light on these qualitative differences:

**Valley Asymmetry:**

**full_pgd**: `634.62`

**decay_control**: `4282.78`

A lower valley asymmetry suggests that the local geometry of the loss basin is more symmetric. In practical terms, **full_pgd**'s smaller asymmetry indicates a more consistent "valley shape," which often translates to more uniform performance under random parameter perturbations. By contrast, the higher asymmetry in **decay_control** suggests deeper or more irregular pockets, potentially making the model's behavior more sensitive to noise.

**Top Eigenvalues of the Hessian:**

**full_pgd**: $[1.0379 \times 10^9, 3.06 \times 10^7, 2.45 \times 10^7]$

**decay_control**: $[1.596 \times 10^8, 3.08 \times 10^7, 1.15 \times 10^7]$

Large top eigenvalues signal directions of steep _curvature_ in parameter space. **full_pgd** has a larger principal eigenvalue (~10⁹), suggesting at least one direction of very steep curvature. However, combined with the lower valley asymmetry, this may imply that while there is a steep direction, the overall basin is still more "rounded"." **decay_control**'s principal eigenvalue is smaller, yet it has a bigger gap between eigenvalues, and its landscape is more asymmetric overall—visible in the more irregular "peak" structures.


**Multi-Scale Sharpness:**

**full_pgd**

$\alpha_{0.1}: 8.856 \times 10^8$
$\alpha_{0.01}: 1.076 \times 10^7$
$\alpha_{0.001}: 5.261 \times 10^3$

**decay_control**

$\alpha_{0.1}: 8.010 \times 10^8$
$\alpha_{0.01}: 3.5225 \times 10^5$
$\alpha_{0.001}: 1.040 \times 10^3$

Multi-scale sharpness measures how much the loss can change under parameter perturbations of varying magnitudes **α**. Both models exhibit large values at **α**=0.1, but **full_pgd** is higher, indicating a steeper slope in at least one direction at larger perturbations. However, the difference in the smaller **α** scales (0.01 and 0.001) indicate that **full_pgd** remains more stable than **decay_control** as we zoom in closer to the parameter optimum, hence the fairly high **α₀.₀₁** but simultaneously a lower asymmetry measure.


**α-Sharpness:**

**full_pgd**: `6.0435×10⁸`

**decay_control**: `5.275×10⁸`

The **α**-sharpness metric shows similar trends at a single scale, with **full_pgd** being slightly sharper overall. This again suggests a steep boundary in certain directions.

### Geometry & Noise Robustness

These geometrical differences may help explain why **full_pgd** showed better invalid-recall and fewer performance drops under added noise. The lower valley asymmetry implies that small random parameter shifts (such as those induced by noisy gradients or noise in inputs) do not bounce the model into dramatically different loss regions. Consequently, the model more reliably maintains its abstention behavior on invalid inputs.
Meanwhile, the **decay_control** model's high asymmetry and more uneven curvature can lead to larger variability when exposed to noise or adversarial conditions. 


In summary, while **full_pgd** exhibits some steep curvature directions (reflected in the large top eigenvalues), it maintains a more regular overall basin (seen in its low valley asymmetry). This shape likely underpins the model's ability to stay robust—particularly on invalid inputs—when faced with noisy perturbations. By contrast, **decay_control** model shows higher asymmetry and more sharply varied local shapes, contributing to reduced consistency under noise, which aligns with its weaker empirical performance across progressive noise testing.

## Implications For AI Safety

This experiment explored an approach to AI safety using parameter-space interventions through Parameter-based Projected Gradient Descent (**PGD**). While the experimental domain is arithmetic computation, the results suggest possibilities that could be relevant to broader AI safety challenges.

The main finding from the experiments is that parameter-space interventions using **PGD** showed strong results in preventing undesired outputs. Notably, in the arithmetic testbed, both the **pgd** and **full_pgd** models achieved significantly better invalid recall compared to a control model trained with traditional adversarial training, suggesting that operating directly on network parameters might offer advantages over traditional input-based adversarial training approaches. This success in a simple domain points to interesting possibilities for more complex systems.

If this approach could be successfully scaled to more complex systems like Large Language Models (**LLMs**), it might offer new ways to address safety challenges. For instance, rather than relying solely on input filtering or output censoring to prevent the generation of harmful content, parameter-space interventions could potentially create more fundamental barriers against generating such outputs. This could also help address core challenges with current _LLMs_: rather than hallucinating or confabulating answers when uncertain, models might learn to robustly recognize and abstain from making predictions beyond their capability boundaries - similar to how the arithmetic models in this experiment learned to abstain when faced with invalid computations rather than producing incorrect outputs.

This approach could also help manage the behaviour of potential future advanced AI systems. While currently working in a simplified test case, the results suggest that we might be able to develop AI systems that are fundamentally less prone to concerning behaviors - things like extreme self-preservation drives, power-seeking, or deception. Instead of trying to control these behaviors after they emerge, by tweaking parameters during development we could potentially shape the 'character' of these systems from the ground up, and _developmentally_ align their values with our own. 

One advantage of this approach is its potential for rigorous testing - when we modify parameters, we can measure and analyze the resulting changes in model behavior in a systematic way. While initial experiments in the arithmetic domain showed some encouraging results compared to traditional adversarial training, it is unclear if this technique could be scaled to even simple transformer architecture models, given their richer and more complex latent space. 

That said, I believe this line of research could contribute meaningfully to the broader field of AI safety. As we work toward developing safer and more capable AI systems, the ability to potentially shape an AI's underlying behavioral tendencies through parameter-space modifications might prove to be a valuable tool in our toolkit - even as just one technique used in conjuction with others. 

## Overall Takeaways

**Robustness Gains:** The parameter-based **PGD** technique showed promise in making models more robust to noisy inputs, particularly in encouraging correct abstention on invalid cases.

**Balanced Performance:** Although improvements on the OOD and boundary tests were more modest, the overall trends and statistically significant gains in noise robustness are suggestive of the techniques efficacy. 

**Future Directions:** The mixed results on certain test sets open avenues for further investigation, such as tweaking **PGD** hyper-parameters or exploring hybrid approaches that combine both input and parameter-space adversarial training. Attempts to incorporate parameter based **PGD** training with basic transformer models, along with improvements to the technique to reduce variation in performance across seeds, are two directions I'd be interested in seeing explored further. 
