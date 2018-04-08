# Analysis of van Dokkum et. al. 2018 

An analysis of the star cluster velocities published in van Dokkum et. al 2018 (https://arxiv.org/abs/1803.10237) using `python` and `pymc3`. This was prompted by [this](https://twitter.com/nfmartin1980/status/982245161735372804) twitter thread, by the astronomers [Nicholas Martin](https://twitter.com/nfmartin1980) and [Nicholas Longeard](https://twitter.com/Nico_Longeard).

I find the most probable value of $\sigma=10.3^{+5.7}\_{-4.1}$ km/s, with the posterior shown in `Plots/pdf.pdf`. Allowing for some of the data to be classed as an "outlier", as in the excellent paper Hogg, Bovy and Lang 2010 (https://arxiv.org/pdf/1008.4686) leads to a very similar posterior (with the most probable $\sigma=8.7 ^{+6.0}\_{-4.2}$ km/s). This posterior is shown in `Plots/pdf_outliers.pdf`.

You can read some of my thoughts about this paper and my analysis here: https://medium.com/@samvaughan01/a-galaxy-without-dark-matter-ae18003b87c

![PDF of Sigma](https://github.com/samvaughan/van-Dokkum-2018-analysis/blob/master/pdf.jpg)
