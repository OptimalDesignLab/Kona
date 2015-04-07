import kona

solver = UserTemplate(2,2,2)

options = {}

optObj = kona.Optimizer(solver, options)

optObj.Optimize()