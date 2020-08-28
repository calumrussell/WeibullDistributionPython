Implements the Weibull Count discrete distribution from [McShane, 2008] (http://www.blakemcshane.com/Papers/jbes_weibull.pdf). The basic implementation is partly taken from [Countr] (https://github.com/GeoBosh/Countr) but we have forgone some optimisations in order to improve the readability of the code. In particular, where possible, the most important operations have been vectorized so the logic is clear.

The Weibull Count distribution takes a location and a scale parameter. It can be thought of as extension of the Poisson where a Weibull Count with a scale parameter of 1 will be equal to a Poisson with the same location parameter (and this is how we have tested our code).

Our implementation also takes a precision and outcomes parameter. Outcomes is simply the number of discrete outcomes that you wish to compute. Precision controls the precision of the probability estimate for a given discrete outcome. It is important to note that increasing both of these parameters will affect running time. Increasing outcomes marginally affects running times, as it adds operations only at the outermost loop. Increasing precision affects running times to a greater extent as it will add operations in the innermost loop. The default precision parameter should be effective for most use-cases.

We have attempted to cover the main operations of the distribution as scipy. The only different design choice, which was a function of our application, was to pass the location and scale parameters to the distribution object on creation.

Finally, bear in mind that the creation of random values is somewhat slower than scipy once size passes ~10m. Some effort has been made to optimise this operation but is limited by the speed of Python loops.
