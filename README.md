# LoRA-FineTuning-FNN
Low Rank Adaption on MNIST data<br>
LoRA hypothesis : when adapting to a specific task it is shown that pre trained LM have a low intrinsic dimension <br>
and can still learn efficiently despite a ranodm porjection to a smaller subspace thats why it is hypothesized that <br>
updatation to weights also have less intrinsic rank during adaption. <br>
h = WoX + (dW)X = WoX + (BA)X<br>
 We then scale ∆Wx by α/r , where α is a constant in r.
