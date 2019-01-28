# Foreword

I find AI Safety via Debate [1] to be a super interesting article. I love how the authors take an important problem, translate it into a tractable one and solve it via a simple, but powerful idea. All the while still being able to draw some theoretical insights about the original goal.

# My summary of the paper

The authors are concerned with AI alignment which is a big topic in AI safety, as we would like to use AI for solving our goals, but it is unclear how to align actions of an AI which we are unable to follow because it has superhuman capabilities.

A protocol is proposed where 2 agents play a game of debate in which they try to convince a judge of their claims about a given problem. The judge has a limited capability to solve the problem compared to the agents. The catch is that only one of the agents is honest. The other one is adversarial and tries to convince the judge of a lie.

A practical example is evaluated in the paper: The judge is a ML-based classifier of MNIST digits. Judge’s limited capability is simulated by allowing it to see only 6 pixels from the whole image, while the agents can see the whole image. The game starts by the agents pre-committing to a category they will argue for. They then take turns to show a single pixel to the judge. At the end of the game the judge announces the verdict which of the two pre-committed classes is more likely given the 6 pixels as evidence. This protocol increases the accuracy of the judge from 59.4% when solving the problem alone to 87.7%.

Debate is then further analysed from perspective of complexity theory, as well as an interesting discussion is provided about different variants of the game and its impact on the AI alignment problem. But I will stop here with the summary and try to describe my remarks.

# My remarks & about this project

I argue that much of the judge's accuracy gain is due to the pre-commit itself and not due to the debate between the agents. Choosing from 2 categories instead of 10 is much easier. To support my claim I am empirically showing in this project that a gain in accuracy (55.51% to 88.3%) similar to that in the paper can be obtained with only the pre-commit, without the actual debate. This is when the two pre-committed classes are the true and another randomly chosen class.

However, I must admit that an adversarially chosen lie can substantially decrease the accuracy. In my tests I observed the accuracy drop to 75.41%. In the limit case of an adversary with complete knowledge of the judge it's even possible to remove the gain in accuracy completely.

# Conclusion

I showed that much of the gain in accuracy can be explained solely by the pre-commit. Nonetheless, the game of debate seems to me as a good tool for finding the candidate solutions (values for pre-commit) via agents of superior capabilities, and for mitigating the negative effect of the adversary.

# Future work

It's a question whether we can talk about honest and adversarial agents in the context of AI safety. Either we accept that agents may generally want to deceive us, then we can't assume even one honest agent. Or we can decide to assume both of the agents are honest. But even if both agents are acting in a good faith there will be cases of disagreement. This is a known problem usually solved by ensembling methods. Could the game of debate be used as an addition to the existing ensembling methods? If you want to see some preliminary exploration of this idea, take a look at the future_work.ipynb notebook.

# References

\[1\] [AI safety via debate](https://arxiv.org/abs/1805.00899v2), Irving, Geoffrey; Christiano, Paul; Amodei, Dario, 2018