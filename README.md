LDPUTS is a Python package that provides simple implementation and simulation of non-parametric two-sample testing under local differential privacy based on U-statistic and permutation test.
The primary objective of this project is to offer a user-friendly interface for conducting benchmarking and experimentation with the privacy mechanism and testing procedure. However, it is important to note that this package is not intended for practial development; This package violates some ciritical assmuptions of deploymnent of local differential privacy.
For example, the raw dataset is first loaded in its entirety on the client side before to the implementation of privacy 
mechanism. 
This practice deviates from the fundamental assumption that each individual data point within the raw dataset can only be accessed by its own data owner, while the curator is limited to observing only the privatized releases.
This violation pertains to the implementation of efficient discretization and privatization for large-scale datasets.

Although local differential privacy is based on individual data owners

If pure-LDP is useful to you and has been used in your work in any way we would appreciate a reference to: