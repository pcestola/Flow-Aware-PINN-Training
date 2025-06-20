# Progressive Domain Training for Physics-Informed Neural Networks (PINNs)

![Poisson 2D - C](poisson_2d_c.gif)
![Poisson 2D - CG](poisson_2d_cg.gif)

> This repository contains the official implementation of the experiments from  
> **"A Progressive Domain Decomposition Strategy for Training Physics-Informed Neural Networks"**  
> by Pietro Cestola (2025)

---

## Summary

This project introduces a novel training strategy for **Physics-Informed Neural Networks (PINNs)** that aligns the optimization process with the **natural flow of physical information**. Instead of training the model over the full domain from the start, we propose a **progressive domain decomposition** approach, where the training region grows over time from boundary-constrained regions to the interior.

---

## Motivation

Traditional PINN training does not reflect the causal structure of the physical system. By **exploiting the directional propagation of information**, our method improves:

- convergence speed
- computational efficiency
- solution accuracy

---

## Key Features

- Progressive domain expansion using **geodesic distance partitioning**
- Compatible with irregular and multi-hole 2D geometrie

---

## Examples

<table>
  <tr>
    <td align="center">
      <b>Wave 1D</b><br/>
      Symmetric propagation of a Gaussian pulse over time.<br/>
      <img src="images/gifs/output_ini_sol.gif" width="250"/>
    </td>
    <td align="center">
      <b>Poisson 2D – C</b><br/>
      Rectangular domain with four circular holes.<br/>
      <img src="images/gifs/poisson_2d_c.gif" width="250"/>
    </td>
    <td align="center">
      <b>Poisson 2D – CG</b><br/>
      Irregular domain with oscillatory source term and inclusions.<br/>
      <img src="images/gifs/poisson_2d_cg.gif" width="250"/>
    </td>
  </tr>
</table>


---
