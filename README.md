# Progressive Domain Training for Physics-Informed Neural Networks (PINNs)

## Summary

This project introduces a novel training strategy for **Physics-Informed Neural Networks (PINNs)** that aligns the optimization process with the **natural flow of physical information**. Instead of training the model over the full domain from the start, we propose a **progressive domain decomposition** approach, where the training region grows over time from boundary-constrained regions to the interior.

## Examples

<table>
  <tr>
    <td align="center">
      <b>Wave 1D</b><br/>
      Symmetric propagation of a Gaussian pulse over time.<br/>
      <img src="images/gifs/wave_1d.gif" width="250"/>
    </td>
    <td align="center">
      <b>Poisson 2D – C</b><br/>
      Rectangular domain with four circular holes.<br/>
      <img src="images/gifs/poisson_2d_c.gif" width="250"/>
    </td>
  </tr>
</table>


---
