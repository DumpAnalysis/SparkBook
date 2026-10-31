


def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    mul = l_x_ * 1.25;  l_x_ = None
    add = mul + 0.5;  mul = None
    relu = torch.relu(add);  add = None
    return (relu,)
    